
import argparse
import logging
import re
import typing


if typing.TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
else:
    BaseRetriever = typing.Any

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)

LOG = logging.getLogger(__name__)


def process_prompt(retriever: BaseRetriever, prompt: str):
    # Show "Next document" prompt if prompt ends with '..'
    auto_pause = prompt[-2:] == ".."
    prompt = re.sub(r"(\.\.)$", '', prompt)

    results = retriever.invoke(prompt)
    LOG.info("Got %d documents.", len(results))
    for i, doc in enumerate(results):
        print("=== Document %d ===" % i)
        print("\tPage number: %s\n\tType: %s" % (
            doc.metadata.get("page_number", "Unknown"),
            doc.metadata.get("type", "text"),
        ))
        print(doc.page_content)
        print()
        if auto_pause:
            k = input("Press <Enter> to show next, 'q' and <Enter> to skip.")
            if k.lower() == 'q':
                break


def repl(retriever: BaseRetriever):
    print("Enter 'q', 'exit' or press Ctrl+C to exit.")
    while True:
        try:
            prompt = input("> ")
            if prompt.lower() in ["q", "exit"]:
                break
            process_prompt(retriever, prompt)
        except KeyboardInterrupt:
            print()
            break
        except Exception:
            LOG.exception("Exception while processing prompt:")
    LOG.info("Stopping...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document", help="Document file to process")
    parser.add_argument("-c", "--cache-dir", required=False, default=None)
    parser.add_argument("-q", "--query", help="Run a query and exit", required=False, default=None)
    parser.add_argument("-n", "--document-count", help="Always return at most this number of results", required=False, default=3)

    args = parser.parse_args()

    LOG.info("Loading dependencies")
    from .store import RetrievalStore
    from .retrieval_qa import ingest_process

    LOG.info("Starting ingest process of \"%s\"...", args.document)
    store = RetrievalStore(
        k=args.document_count
    )
    retriever = ingest_process(
        args.document,
        store,
        cache_directory=args.cache_dir
    )

    if args.query is not None:
        process_prompt(retriever, args.query)
    else:
        repl(retriever)
