import argparse
import asyncio
import os
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureDeveloperCliCredential
from dotenv import load_dotenv
from prepdocslib.blobmanager import BlobManager
from prepdocslib.embeddings import (
    AzureOpenAIEmbeddingService,
    OpenAIEmbeddings,
    OpenAIEmbeddingService,
)
from prepdocslib.filestrategy import DocumentAction, FileStrategy
from prepdocslib.listfilestrategy import (
    ADLSGen2ListFileStrategy,
    ListFileStrategy,
    LocalListFileStrategy,
)
from prepdocslib.pdfparser import DocumentAnalysisPdfParser, PdfParser
from prepdocslib.strategy import SearchInfo, Strategy
from prepdocslib.textsplitter import TextSplitter

load_dotenv()


def is_key_empty(key):
    return key is None or len(key.strip()) == 0


def setup_file_strategy(args: Any) -> FileStrategy:
    storage_creds = args.storagekey
    blob_manager = BlobManager(
        endpoint=f"https://{args.storageaccount}.blob.core.windows.net",
        container=args.container,
        credential=storage_creds,
        verbose=args.verbose,
    )

    pdf_parser: PdfParser
    # check if Azure Form Recognizer credentials are provided
    if args.formrecognizerservice is None:
        print(
            "Error: Azure Form Recognizer service is not provided. Please provide formrecognizerservice"
        )
        exit(1)
    formrecognizer_creds = AzureKeyCredential(args.formrecognizerkey)
    pdf_parser = DocumentAnalysisPdfParser(
        endpoint=f"https://{args.formrecognizerservice}.cognitiveservices.azure.com/",
        credential=formrecognizer_creds,
        verbose=args.verbose,
    )

    use_vectors = not args.novectors
    embeddings: Optional[OpenAIEmbeddings] = None
    if use_vectors and args.openaihost != "openai":
        azure_open_ai_credential: Union[
            AsyncTokenCredential, AzureKeyCredential
        ] = AzureKeyCredential(os.getenv("OPENAI_API_KEY"))
        embeddings = AzureOpenAIEmbeddingService(
            open_ai_service=args.openaiservice,
            open_ai_deployment=args.openaideployment,
            open_ai_model_name=args.openaimodelname,
            credential=azure_open_ai_credential,
            disable_batch=args.disablebatchvectors,
            verbose=args.verbose,
        )

    print("Processing files...")
    list_file_strategy: ListFileStrategy

    print(f"Using local files in {args.files}")
    list_file_strategy = LocalListFileStrategy(
        path_pattern=args.files, verbose=args.verbose
    )

    if args.removeall:
        document_action = DocumentAction.RemoveAll
    elif args.remove:
        document_action = DocumentAction.Remove
    else:
        document_action = DocumentAction.Add

    return FileStrategy(
        list_file_strategy=list_file_strategy,
        blob_manager=blob_manager,
        pdf_parser=pdf_parser,
        text_splitter=TextSplitter(),
        document_action=document_action,
        embeddings=embeddings,
        search_analyzer_name=args.searchanalyzername,
        use_acls=args.useacls,
        category=args.category,
    )


async def main(strategy: Strategy, args: Any):
    search_creds = AzureKeyCredential(args.searchkey)
    search_info = SearchInfo(
        endpoint=f"https://{args.searchservice}.search.windows.net/",
        credential=search_creds,
        index_name=args.index,
        verbose=args.verbose,
    )

    if not args.remove and not args.removeall:
        await strategy.setup(search_info)

    await strategy.run(search_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare documents by extracting content from PDFs, splitting content into sections, uploading to blob storage, and indexing in a search index.",
        epilog="Example: prepdocs.py '..\data\*' --storageaccount myaccount --container mycontainer --searchservice mysearch --index myindex -v",
    )
    parser.add_argument("files", nargs="?", help="Files to be processed")
    parser.add_argument(
        "--useacls",
        action="store_true",
        help="Skip uploading individual pages to Azure Blob Storage",
    )
    parser.add_argument(
        "--storageaccount",
        default="dtfsdocumentchatstorage",
        help="Azure Blob Storage account name",
    ),
    parser.add_argument(
        "--category",
        help="Value for the category field in the search index for all sections indexed in this run",
    )
    parser.add_argument(
        "--container", default="devcontent", help="Azure Blob Storage container name"
    )
    parser.add_argument(
        "--storagekey",
        default=os.getenv("BLOB_STORAGE_KEY"),
        help="Optional. Use this Azure Blob Storage account key instead of the current user identity to login (use az login to set current user for Azure)",
    )
    parser.add_argument(
        "--tenantid",
        required=False,
        help="Optional. Use this to define the Azure directory where to authenticate)",
    )
    parser.add_argument(
        "--searchservice",
        default=os.getenv("AZURE_SEARCH_SERVICE"),
        help="Name of the Azure Cognitive Search service where content should be indexed (must exist already)",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("AZURE_SEARCH_INDEX"),
        help="Name of the Azure Cognitive Search index where content should be indexed (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--searchkey",
        default=os.getenv("AZURE_COGNITIVE_SEARCH_ADMIN_KEY"),
        help="Optional. Use this Azure Cognitive Search account key instead of the current user identity to login (use az login to set current user for Azure)",
    )
    parser.add_argument(
        "--searchanalyzername",
        required=False,
        default="en.microsoft",
        help="Optional. Name of the Azure Cognitive Search analyzer to use for the content field in the index",
    )
    parser.add_argument(
        "--openaihost",
        default="azure",
        help="Host of the API used to compute embeddings ('azure' or 'openai')",
    )
    parser.add_argument(
        "--openaiservice",
        default="DTFS-DocumentChat-OpenAI-Sweden",
        help="Name of the Azure OpenAI service used to compute embeddings",
    )
    parser.add_argument(
        "--openaideployment",
        default="ada2",
        help="Name of the Azure OpenAI model deployment for an embedding model ('text-embedding-ada-002' recommended)",
    )
    parser.add_argument(
        "--openaimodelname",
        default="text-embedding-ada-002",
        help="Name of the Azure OpenAI embedding model ('text-embedding-ada-002' recommended)",
    )
    parser.add_argument(
        "--novectors",
        action="store_true",
        help="Don't compute embeddings for the sections (e.g. don't call the OpenAI embeddings API during indexing)",
    )
    parser.add_argument(
        "--disablebatchvectors",
        action="store_true",
        help="Don't compute embeddings in batch for the sections",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove references to this document from blob storage and the search index",
    )
    parser.add_argument(
        "--removeall",
        action="store_true",
        help="Remove all blobs from blob storage and documents from the search index",
    )
    parser.add_argument(
        "--formrecognizerservice",
        default="dtfs-document-intelligence",
        help="Optional. Name of the Azure Form Recognizer service which will be used to extract text, tables and layout from the documents (must exist already)",
    )
    parser.add_argument(
        "--formrecognizerkey",
        default=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        help="Optional. Use this Azure Form Recognizer account key instead of the current user identity to login (use az login to set current user for Azure)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(args)
    file_strategy = setup_file_strategy(args)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(file_strategy, args))
    loop.close()
