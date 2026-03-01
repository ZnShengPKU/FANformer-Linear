from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.utils.data


def get_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


class BaseStreamingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        paths: Iterable[str],
        input_ids_field: str = "input_ids",
        text_field: Optional[str] = None,
        tokenizer_identifier: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        add_eos: bool = True,
        start_index: int = 0,
        epoch: int = 0,
        exclude_paths: Optional[Iterable[str]] = None,
    ) -> None:
        self.paths = list(paths)
        self.input_ids_field = input_ids_field
        self.text_field = text_field
        self.tokenizer_identifier = tokenizer_identifier
        self.eos_token_id = eos_token_id
        self.max_sequence_length = max_sequence_length
        self.add_eos = add_eos
        self.start_index = start_index
        self.epoch = epoch
        self._tokenizer = None
        self._exclude_paths = {str(Path(path).resolve()) for path in (exclude_paths or []) if path is not None}

    def reshuffle(self, epoch: int) -> None:
        self.epoch = epoch

    def _iter_records(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def _is_excluded(self, path: str) -> bool:
        if not self._exclude_paths:
            return False
        return str(Path(path).resolve()) in self._exclude_paths

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.tokenizer_identifier is None:
            raise ValueError("tokenizer_identifier is required when text_field is set")
        try:
            from tokenizers import Tokenizer as BaseTokenizer
        except Exception as exc:
            raise ValueError("tokenizers is required for text tokenization") from exc
        tokenizer_identifier = self.tokenizer_identifier
        tokenizer_path = Path(tokenizer_identifier)
        if tokenizer_path.is_dir():
            tokenizer_json = tokenizer_path / "tokenizer.json"
            if tokenizer_json.is_file():
                tokenizer = BaseTokenizer.from_file(str(tokenizer_json))
            else:
                tokenizer = BaseTokenizer.from_pretrained(tokenizer_identifier)
        elif tokenizer_path.is_file():
            tokenizer = BaseTokenizer.from_file(str(tokenizer_identifier))
        else:
            tokenizer = BaseTokenizer.from_pretrained(tokenizer_identifier)
        self._tokenizer = tokenizer
        return tokenizer

    def _iter_examples(self) -> Iterable[Dict[str, Any]]:
        if self.text_field:
            if self.max_sequence_length is None:
                raise ValueError("max_sequence_length is required when text_field is set")
            if self.eos_token_id is None:
                raise ValueError("eos_token_id is required when text_field is set")
            tokenizer = self._get_tokenizer()
            buffer: List[int] = []
            for record in self._iter_records():
                text = record.get(self.text_field)
                if not text:
                    continue
                encoded = tokenizer.encode(text)
                ids = list(encoded.ids)
                if self.add_eos:
                    ids.append(self.eos_token_id)
                buffer.extend(ids)
                while len(buffer) >= self.max_sequence_length:
                    chunk = buffer[: self.max_sequence_length]
                    buffer = buffer[self.max_sequence_length :]
                    yield {"input_ids": chunk}
        else:
            for record in self._iter_records():
                value = record.get(self.input_ids_field)
                if value is None:
                    continue
                if isinstance(value, str):
                    value = [int(x) for x in value.split()]
                if self.max_sequence_length:
                    start = 0
                    while len(value) - start >= self.max_sequence_length:
                        chunk = value[start : start + self.max_sequence_length]
                        start += self.max_sequence_length
                        yield dict(record, input_ids=chunk)
                else:
                    yield dict(record, input_ids=value)

    def __iter__(self):
        rank = get_global_rank()
        world_size = get_world_size()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        shard_count = world_size * num_workers
        shard_id = rank * num_workers + worker_id

        index = 0
        for record in self._iter_examples():
            if index < self.start_index:
                index += 1
                continue
            if index % shard_count != shard_id:
                index += 1
                continue
            yield record
            index += 1


class JsonlZstStreamingDataset(BaseStreamingDataset):
    def _iter_records(self) -> Iterable[Dict[str, Any]]:
        try:
            import zstandard as zstd
        except Exception as exc:
            raise ValueError("zstandard is required to read .jsonl.zst files") from exc

        for path in self.paths:
            if Path(path).is_dir():
                entries = sorted(list(Path(path).glob("*.jsonl.zst")) + list(Path(path).glob("*.jsonl")))
                for entry in entries:
                    if self._is_excluded(str(entry)):
                        continue
                    yield from self._read_jsonl(str(entry), zstd)
                continue
            if self._is_excluded(path):
                continue
            yield from self._read_jsonl(path, zstd)

    def _read_jsonl(self, path: str, zstd_module) -> Iterable[Dict[str, Any]]:
        if path.endswith(".zst"):
            with open(path, "rb") as file_handle:
                dctx = zstd_module.ZstdDecompressor()
                with dctx.stream_reader(file_handle) as reader:
                    text_stream = io.TextIOWrapper(reader)
                    for line in text_stream:
                        if not line:
                            continue
                        yield json.loads(line)
        else:
            with open(path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if not line:
                        continue
                    yield json.loads(line)


class ParquetStreamingDataset(BaseStreamingDataset):
    def __init__(
        self,
        paths: Iterable[str],
        input_ids_field: str = "input_ids",
        text_field: Optional[str] = None,
        tokenizer_identifier: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        add_eos: bool = True,
        start_index: int = 0,
        epoch: int = 0,
        batch_size: int = 1024,
        exclude_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(
            paths,
            input_ids_field=input_ids_field,
            text_field=text_field,
            tokenizer_identifier=tokenizer_identifier,
            eos_token_id=eos_token_id,
            max_sequence_length=max_sequence_length,
            add_eos=add_eos,
            start_index=start_index,
            epoch=epoch,
            exclude_paths=exclude_paths,
        )
        self.batch_size = batch_size

    def _iter_records(self) -> Iterable[Dict[str, Any]]:
        try:
            import pyarrow.dataset as ds
        except Exception as exc:
            raise ValueError("pyarrow is required to read .parquet files") from exc

        for path in self.paths:
            if Path(path).is_dir():
                for entry in sorted(Path(path).glob("*.parquet")):
                    if self._is_excluded(str(entry)):
                        continue
                    yield from self._iter_parquet(str(entry), ds)
                continue
            if self._is_excluded(path):
                continue
            yield from self._iter_parquet(path, ds)

    def _iter_parquet(self, path: str, ds_module) -> Iterable[Dict[str, Any]]:
        dataset = ds_module.dataset(path, format="parquet")
        column_name = self.text_field or self.input_ids_field
        scanner = dataset.scanner(columns=[column_name], batch_size=self.batch_size)
        for batch in scanner.to_batches():
            column = batch.column(column_name).to_pylist()
            for value in column:
                if value is None:
                    continue
                yield {column_name: value}


class MultiFormatStreamingDataset(BaseStreamingDataset):
    def __init__(
        self,
        paths: Iterable[str],
        input_ids_field: str = "input_ids",
        text_field: Optional[str] = None,
        tokenizer_identifier: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        add_eos: bool = True,
        start_index: int = 0,
        epoch: int = 0,
        batch_size: int = 1024,
        exclude_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(
            paths,
            input_ids_field=input_ids_field,
            text_field=text_field,
            tokenizer_identifier=tokenizer_identifier,
            eos_token_id=eos_token_id,
            max_sequence_length=max_sequence_length,
            add_eos=add_eos,
            start_index=start_index,
            epoch=epoch,
            exclude_paths=exclude_paths,
        )
        self.batch_size = batch_size

    def _iter_records(self) -> Iterable[Dict[str, Any]]:
        for path in self.paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                entries = sorted(
                    list(path_obj.glob("*.jsonl.zst"))
                    + list(path_obj.glob("*.jsonl"))
                    + list(path_obj.glob("*.parquet"))
                )
                for entry in entries:
                    if self._is_excluded(str(entry)):
                        continue
                    yield from self._iter_single_path(str(entry))
                continue
            if self._is_excluded(path):
                continue
            yield from self._iter_single_path(path)

    def _iter_single_path(self, path: str) -> Iterable[Dict[str, Any]]:
        if path.endswith(".parquet"):
            dataset = ParquetStreamingDataset(
                [path],
                input_ids_field=self.input_ids_field,
                text_field=self.text_field,
                tokenizer_identifier=self.tokenizer_identifier,
                eos_token_id=self.eos_token_id,
                max_sequence_length=self.max_sequence_length,
                add_eos=self.add_eos,
                start_index=self.start_index,
                epoch=self.epoch,
                batch_size=self.batch_size,
            )
            yield from dataset._iter_records()
            return
        dataset = JsonlZstStreamingDataset(
            [path],
            input_ids_field=self.input_ids_field,
            text_field=self.text_field,
            tokenizer_identifier=self.tokenizer_identifier,
            eos_token_id=self.eos_token_id,
            max_sequence_length=self.max_sequence_length,
            add_eos=self.add_eos,
            start_index=self.start_index,
            epoch=self.epoch,
        )
        yield from dataset._iter_records()
