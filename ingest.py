from functools import wraps
import hashlib
from os import PathLike
from pathlib import Path
import pickle as pkl
from typing import Optional
from unstructured.partition.auto import partition
import store


def pickle_cache(f):
    @wraps(f)
    def _wrapper(*args, cache_directory: Optional[PathLike] = None, **kwargs):
        if cache_directory is not None:
            cache_directory = Path(cache_directory)
            if not cache_directory.is_dir():
                cache_directory.mkdir(parents=True, exist_ok=True)

            cache_id = hashlib.sha256((str(args)+str(kwargs)).encode()).hexdigest()
            cache_file = cache_directory.joinpath(cache_id).with_suffix(".pkl")

            if cache_file.exists():
                # Cache hit
                with open(cache_file, "rb") as inp_f:
                    return pkl.load(inp_f)

            # Cache miss
            # Call and save response to cache file
            obj = f(*args, **kwargs)
            with open(cache_file, "wb") as op_f:
                pkl.dump(obj, op_f)
            return obj

        # No caching
        return f(*args, **kwargs)
    return _wrapper


def parse_document_unstructured(document_file: Path, **options):
    opts = dict(
        filename=document_file,

        skip_infer_table_types=[],
        # Use layout model (YOLOX) to get bounding boxes (for tables)
        # and find titles
        # Titles are any sub-section of the document
        pdf_infer_table_structure=True,

        # Unstructured first finds embedded image blocks
        extract_images_in_pdf=False,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        max_characters=480,  # 4000,
        new_after_n_chars=400,  # 3800,
        combine_text_under_n_chars=380,  # 2000,
        include_page_breaks=True,
    )

    opts.update(options)

    return partition(**opts)


@pickle_cache
def ingest_document(
    document_file: Path, **parser_options
):
    # Cast to Path if it's not
    document_file = Path(document_file)

    return parse_document_unstructured(document_file, **parser_options)


def get_retriever():
    return store.retriever
