
import argparse
import logging
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


def process_prompt(retriever: BaseRetriever, prompt: str, auto_pause: bool = False):
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
            input("Press <Enter> to show next.")

def repl(retriever: BaseRetriever):
    print("Enter 'q', 'exit' or press Ctrl+C to exit.")
    while True:
        try:
            prompt = input("> ")
            if prompt.lower() in ["q", "exit"]:
                break
            auto_pause = prompt[-2:] == ".."
            process_prompt(retriever, prompt, auto_pause)
        except KeyboardInterrupt:
            print()
            break
        except:
            LOG.exception("Exception while processing prompt:")
    LOG.info("Stopping...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document", help="Document file to process")
    parser.add_argument("-q", "--cache-dir", required=False, default=None)

    args = parser.parse_args()

    LOG.info("Loading dependencies")    
    from .store import RetrievalStore
    from .retrieval_qa import ingest_process

    LOG.info("Starting ingest process of \"%s\"...", args.document)
    store = RetrievalStore()
    retriever = ingest_process(
        args.document,
        store,
        cache_directory=args.cache_dir
    )
    repl(retriever)
