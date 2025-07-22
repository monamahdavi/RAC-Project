# KNN
from typing import List
from uuid import uuid4

from langchain.schema import Document
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
from tqdm import tqdm
from chromadb.config import Settings
import logging 
for handler in logging.root.handlers [:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

logger.info("Hello World")


class DatasetVectorStore:
    """ChromaDB vector store for PublicationModel objects with SentenceTransformers embeddings."""

    def __init__(
        self,
        db_name: str = "retrieval_augmented_classification",  # Using db_name as collection name in Chroma
        collection_name: str = "classification_dataset",
        persist_directory: str = "chroma_db",  # Directory to persist ChromaDB
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Determine if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={
                "device": device,
                "batch_size": 100,
            },  # Adjust batch_size as needed
        )

        # Initialize Chroma vector store
        self.client = PersistentClient(
            path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        # keep client information privately
        
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List) -> None:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of dictionaries containing document data.  Each dict needs a "text" key.
        """

        local_documents = []
        ids = []
        # make empty list for documents saving

        for doc_data in documents:
            if not doc_data.get("id"):
                doc_data["id"] = str(uuid4())

            # uuind 4 : make a unique id for exmaple: {"text":"I am trying to learn LLMs","category":"training", "id": "a1b2c3..."}

            local_documents.append(
                Document(
                    page_content=doc_data["text"],
                    metadata={k: v for k, v in doc_data.items() if k != "text"},
                )
            )
            ids.append(doc_data["id"])

        batch_size = 100  # Adjust batch size as needed
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = local_documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # Chroma's add_documents doesn't directly support pre-defined IDs. Upsert instead.
            self._upsert_batch(batch_docs, batch_ids)

    def _upsert_batch(self, batch_docs: List[Document], batch_ids: List[str]):
        """Upsert a batch of documents into Chroma.  If the ID exists, it updates; otherwise, it creates."""
        texts = [doc.page_content for doc in batch_docs]
        metadatas = [doc.metadata for doc in batch_docs]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=batch_ids)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search documents by semantic similarity."""
        results = self.vector_store.similarity_search(query, k=3)
        return results

# it work based on semantic search (jostojoo bar asaas ma'na)
# __init__ Function: 
# 1. It sets the name of the database and the path where the data will be saved.
# 2. Then it checks if the system is using CPU or GPU. (GPU is faster for processing.)
# 3. Next, it loads an embedding model that can turn text into numbers.
# 4. This model always turns any sentence into a vector with 384 numbers.
# 5. Finally, it creates a ChromaDB vector database, so that later, it can store the text and its vector together.
# __add__ _documents Function: 
# 1. If the text doesn't have an ID, it creates a unique ID for it.
# 2. It prepares the text and its info (like category).
# 3. It changes the text into numbers (a vector).
# 4. It saves everything into the Chroma database.
# __upsert__ _batch Function: update + insert
#
# This function takes a batch of documents and their IDs, and stores them in ChromaDB.
#
# Steps:
# 1. It separates the actual text from each document into a 'texts' list.
# 2. It collects the extra info (like category and id) into a 'metadatas' list.
# 3. It also creates a separate 'ids' list, even though IDs are already in the metadata.
#
# Why do we separate the IDs?
# - ChromaDB requires 'ids' to be passed separately.
# - It uses them to know whether to insert or update a document (upsert).
# - It also uses the ID as a unique key for searching or managing documents.
#
# Even if the ID is inside the metadata, Chroma doesn't use it unless it's passed directly in the 'ids' list.
#
# Only the 'text' is turned into a vector (embedding). The 'metadata' is just stored along with it for filtering or reference.
# store = DatasetVectorStore()
# RAC
from typing import Optional
from pydantic import BaseModel, Field
from collections import Counter

#from retrieval_augmented_classification.vector_store import DatasetVectorStore (we dont need this command, because all codes are together)
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class PredictedCategories(BaseModel):
    """
    Pydantic model for the predicted categories from the LLM.
    """

    reasoning: str = Field(description="Explain your reasoning")
    predicted_category: str = Field(description="Category")
# This class defines the format of the model's answer: 
# It must return a reason and a predicted category.

class RAC:
    """
    A hybrid classifier combining K-Nearest Neighbors retrieval with an LLM for multi-class prediction.
    Finds top K neighbors, uses top few-shot for context, and uses all neighbor categories
    as potential prediction candidates for the LLM.
    """

    def __init__(
        self,
        vector_store: DatasetVectorStore,
        llm_client,
        knn_k_search: int = 30,
        knn_k_few_shot: int = 5,
    ):
        """
        Initializes the classifier.

        Args:
            vector_store: An instance of DatasetVectorStore with a search method.
            llm_client: An instance of the LLM client capable of structured output.
            knn_k_search: The number of nearest neighbors to retrieve from the vector store.
            knn_k_few_shot: The number of top neighbors to use as few-shot examples for the LLM.
                           Must be less than or equal to knn_k_search.
        """

        self.vector_store = vector_store
        self.llm_client = llm_client
        self.knn_k_search = knn_k_search
        self.knn_k_few_shot = knn_k_few_shot

    # @retry(
    #     stop=stop_after_attempt(3),  # Retry LLM call a few times
    #     # wait=wait_exponential(multiplier=1, min=2, max=5),  # Shorter waits for demo
    # )
    def predict(self, document_text: str) -> Optional[str]:
        """
        Predicts the relevant categories for a given document text using KNN retrieval and an LLM.

        Args:
            document_text: The text content of the document to classify.

        Returns:
            The predicted category
        """
        neighbors = self.vector_store.search(document_text, k=self.knn_k_search)

        all_neighbor_categories = set()
        valid_neighbors = []  # Store neighbors that have metadata and categories
        for neighbor in neighbors:
            if (
                hasattr(neighbor, "metadata")
                and isinstance(neighbor.metadata, dict)
                and "category" in neighbor.metadata
            ):
                all_neighbor_categories.add(neighbor.metadata["category"])
                valid_neighbors.append(neighbor)
            else:
                pass  # Suppress warnings for cleaner demo output

        if not valid_neighbors:
            return None

        category_counts = Counter(all_neighbor_categories)
        ranked_categories = [
            category for category, count in category_counts.most_common()
        ]

        if not ranked_categories:
            return None

        few_shot_neighbors = valid_neighbors[: self.knn_k_few_shot]

        messages = []

        system_prompt = f"""You are an expert multi-class classifier. Your task is to analyze the provided document text and assign the most relevant category from the list of allowed categories.
You MUST only return categories that are present in the following list: {ranked_categories}.
If none of the allowed categories are relevant, return an empty list.
Return the categories by likelihood (more confident to least confident).
Output your prediction as a JSON object matching the Pydantic schema: {PredictedCategories.model_json_schema()}.
"""
        messages.append(SystemMessage(content=system_prompt))

        for i, neighbor in enumerate(few_shot_neighbors):
            messages.append(
                HumanMessage(content=f"Document: {neighbor.page_content}")
            )
            expected_output_json = PredictedCategories(
                reasoning="Your reasoning here",
                predicted_category=neighbor.metadata["category"]
            ).model_dump_json()
            # Simulate the structure often used with tool calling/structured output

            ai_message_with_tool = AIMessage(
                content=expected_output_json,
            )

            messages.append(ai_message_with_tool)

        # Final user message: The document text to classify
        messages.append(HumanMessage(content=f"Document: {document_text}"))

        # Configure the client for structured output with the Pydantic schema
        structured_client = self.llm_client.with_structured_output(PredictedCategories)
        llm_response: PredictedCategories = structured_client.invoke(messages)

        predicted_category = llm_response.predicted_category

        return predicted_category if predicted_category in ranked_categories else None

# * @retry: it is a decorator : automatically retries the predict() function up to 3 times
# if the LLM call fails (e.g., due to a temporary error or timeout).
# Test
# KNN

from sklearn.model_selection import train_test_split
import pandas as pd

print("Script started")
# Load CSV
print("Loading CSV...")
df = pd.read_csv("top_20_categories_subset.csv")
print("CSV loaded. Splitting train/test...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Train/test split done. Preparing training samples...")
#Prepare training samples
samples = [
    {"text": row["text"], "category": row["l3"]}
    for _, row in train_df.iterrows()
]
print(f"Prepared {len(samples)} training samples. Building vector store...")
# Build vector store with training data
store= DatasetVectorStore()
print("Vector store created. Adding documents...")
# save embedding data
store.add_documents(samples)
print("Documents added to vector store. Defining KNN prediction function...")
# Define KNN prediction function


def knn_predict(vector_store, text, k=5):
    results = vector_store.search(text, k=k)
    
    categories = []
    for doc in results:
        if "category" in doc.metadata:
            categories.append(doc.metadata["category"])
    
    if not categories:
        return None
    
    most_common_category = Counter(categories).most_common(1)[0][0]
    return most_common_category

# if in output 1/1 shows it meand 1 batch, we define batchs in 100 samples, so it means model test 100 samples.
#in this case shows 13 batch

#Accuracy
from sklearn.metrics import accuracy_score,precision_score,f1_score,confusion_matrix

print("Preparing test data...")
# Prepare test data
test_texts = test_df["text"].tolist()
true_labels = test_df["l3"].tolist()
print(f"Test data prepared: {len(test_texts)} samples. Running KNN predictions...")
# Predict using knn
predicted_labels = []

for text in test_texts:
    pred = knn_predict(store, text, k=5)
    predicted_labels.append(pred)
print("KNN predictions done. Calculating accuracy...")


precision = precision_score(true_labels, predicted_labels, average='macro')
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
cm = confusion_matrix(true_labels, predicted_labels)

print(f"KNN Accuracy: {accuracy:.2f}")
print(f"KNN precision: {precision:.2f}")
print(f"KNN f1: {f1:.2f}")

# ! pip install langchain transformers torch pydantic chromadb tqdm
from transformers import pipeline

print("Setting up LLM pipeline...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1,
)
print("LLM pipeline ready. Running LLM-only predictions...")
def llm_only_predict(text: str, candidate_labels: List[str]) -> Optional[str]:
    result = classifier(sequences=text, candidate_labels=candidate_labels)
    return result["labels"][0] if result and "labels" in result else None
# تمام دسته‌ها (یونیک) از دیتافریم
all_categories = sorted(df["l3"].unique().tolist())

# پیش‌بینی برای هر متن در تست‌ست با LLM
test_texts = test_df["text"].tolist()
true_labels = test_df["l3"].tolist()

llm_preds = [llm_only_predict(text, all_categories) for text in test_texts]
print("LLM-only predictions done. Calculating metrics...")

accuracy = accuracy_score(true_labels, llm_preds)
precision = precision_score(true_labels, llm_preds, average='macro')
f1 = f1_score(true_labels, llm_preds, average='macro')

print(f"LLM-only Accuracy: {accuracy:.4f}")
print(f"LLM-only Precision: {precision:.4f}")
print(f"LLM-only F1 Score: {f1:.4f}")

class RAC:
    def __init__(self, vector_store, classifier, knn_k_search=20, knn_k_few_shot=5):
        self.vector_store = vector_store
        self.classifier = classifier
        self.knn_k_search = knn_k_search
        self.knn_k_few_shot = knn_k_few_shot

    def predict(self, document_text: str) -> Optional[str]:
        neighbors = self.vector_store.search(document_text, k=self.knn_k_search)

        valid_neighbors = []
        categories = []

        for neighbor in neighbors:
            if hasattr(neighbor, "metadata") and isinstance(neighbor.metadata, dict) and "category" in neighbor.metadata:
                categories.append(neighbor.metadata["category"])
                valid_neighbors.append(neighbor)

        if not valid_neighbors:
            return None

        # فقط دسته‌های یکتا
        unique_categories = list(set(categories))

        # نمونه‌های few-shot
        few_shot_neighbors = valid_neighbors[: self.knn_k_few_shot]

        # ساختن prompt: اول مثال‌ها، بعد سوال
        prompt_parts = ["You are a text classifier. Given a document, choose the most suitable category from the list provided.\n"]

        for i, neighbor in enumerate(few_shot_neighbors):
            prompt_parts.append(f"Example {i+1}:\nText: {neighbor.page_content}\nCategory: {neighbor.metadata['category']}\n")

        prompt_parts.append(f"Now classify this document:\nText: {document_text}")

        full_prompt = "\n".join(prompt_parts)

        # اجرای مدل BART-MNLI با candidate labels از همسایه‌ها
        result = self.classifier(
            sequences=full_prompt,
            candidate_labels=unique_categories,
        )

        return result["labels"][0] if result and "labels" in result else None
print("Setting up RAC hybrid model...")
rac = RAC(vector_store=store, classifier=classifier, knn_k_search=20, knn_k_few_shot=5)
print("RAC model ready. Running RAC predictions...")
predicted_labels = [rac.predict(text) for text in test_texts]
print("RAC predictions done. Calculating metrics...")

# Accuracy metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"RAC Accuracy: {accuracy:.2f}")
print(f"RAC Precision: {precision:.2f}")
print(f"RAC F1: {f1:.2f}")
# In your knn_predict or RAC predict
# print(f"Query: {text}")
# for doc in results:
#     print(f"Retrieved: {doc.page_content[:100]}... -> {doc.metadata['category']}")
# test by new data


new_text_1= " my favorite field to study is creating and visulizing sports images"
new_text_2 = "Cristiano Ronaldo scored a goal for Portugal"



knn_category = knn_predict(store, new_text_1, k=5)
print("KNN predicted category:", knn_category)

rac_category = rac.predict(new_text_1)
print("RAC predicted category:", rac_category)
knn_category = knn_predict(store, new_text_2, k=5)
print("KNN predicted category:", knn_category)

rac_category = rac.predict(new_text_2)
print("RAC predicted category:", rac_category)
# Sources:
# https://docs.google.com/spreadsheets/d/11WD6SiPNaVIQTzbLkU0K0DmDHYN7GLfOVcj3ap3-MdY/edit?pli=1&gid=0#gid=0
# https://microsoft.github.io/generative-ai-for-beginners/#/
# https://github.com/CVxTz/retrieval_augmented_classification
class RAC:
    def __init__(self, vector_store, classifier, predicted_labels):
        self.vector_store = vector_store
        self.classifier = classifier
        self.knn_k_search = predicted_labels

    def predict(self, document_text: str) -> Optional[str]:
        neighbors = self.vector_store.search(document_text, k=self.knn_k_search)

        # Collect categories from neighbors
        categories = []
        for neighbor in neighbors:
            if (
                hasattr(neighbor, "metadata")
                and isinstance(neighbor.metadata, dict)
                and "category" in neighbor.metadata
            ):
                categories.append(neighbor.metadata["category"])

        if not categories:
            return None

        # Deduplicate
        unique_categories = list(set(categories))

        # Use HF pipeline to classify
        result = self.classifier(
            sequences=document_text,
            candidate_labels=unique_categories,
        )

        # Return the top predicted label
        return result["labels"][0] if result and "labels" in result else None

true_labels = test_df["l3"].tolist()
test_texts = test_df["text"].tolist()


predicted_labels = [rac.predict(text) for text in test_texts]

precision = precision_score(true_labels, predicted_labels, average='macro')
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
cm = confusion_matrix(true_labels, predicted_labels)

print(f"RAC Accuracy: {accuracy:.2f}")
print(f"RAC precision: {precision:.2f}")
print(f"RAC f1: {f1:.2f}")

# --- Visualization and Comparison ---
import matplotlib.pyplot as plt

# Collect metrics for each method
methods = ['KNN', 'LLM', 'RAC']
accuracies = []
precisions = []
f1s = []

# KNN metrics (already computed above)
accuracies.append(accuracy_score(true_labels, [knn_predict(store, t, k=5) for t in test_texts]))
precisions.append(precision_score(true_labels, [knn_predict(store, t, k=5) for t in test_texts], average='macro'))
f1s.append(f1_score(true_labels, [knn_predict(store, t, k=5) for t in test_texts], average='macro'))

# LLM metrics (already computed above)
llm_preds = [llm_only_predict(text, all_categories) for text in test_texts]
accuracies.append(accuracy_score(true_labels, llm_preds))
precisions.append(precision_score(true_labels, llm_preds, average='macro'))
f1s.append(f1_score(true_labels, llm_preds, average='macro'))

# RAC metrics (already computed above)
rac_preds = [rac.predict(text) for text in test_texts]
accuracies.append(accuracy_score(true_labels, rac_preds))
precisions.append(precision_score(true_labels, rac_preds, average='macro'))
f1s.append(f1_score(true_labels, rac_preds, average='macro'))

# Plotting
x = range(len(methods))
plt.figure(figsize=(10,6))
plt.bar([i-0.2 for i in x], accuracies, width=0.2, label='Accuracy')
plt.bar(x, precisions, width=0.2, label='Precision')
plt.bar([i+0.2 for i in x], f1s, width=0.2, label='F1 Score')
plt.xticks(x, methods)
plt.ylabel('Score')
plt.title('Comparison of KNN, LLM, and RAC')
plt.legend()
plt.tight_layout()
plt.show()

# --- Metrics and Confusion Matrices ---
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

methods = ['KNN', 'LLM', 'RAC']
all_preds = []
all_metrics = []
all_cms = []

# KNN
knn_preds = [knn_predict(store, t, k=5) for t in test_texts]
all_preds.append(knn_preds)
knn_acc = accuracy_score(true_labels, knn_preds)
knn_prec = precision_score(true_labels, knn_preds, average='macro', zero_division=0)
knn_f1 = f1_score(true_labels, knn_preds, average='macro', zero_division=0)
knn_cm = confusion_matrix(true_labels, knn_preds)
all_metrics.append((knn_acc, knn_prec, knn_f1))
all_cms.append(knn_cm)
print("\nKNN Results:")
print(f"Accuracy: {knn_acc:.2f}")
print(f"Precision: {knn_prec:.2f}")
print(f"F1: {knn_f1:.2f}")
print("Confusion Matrix:\n", knn_cm)

# LLM
llm_preds = [llm_only_predict(text, all_categories) for text in test_texts]
all_preds.append(llm_preds)
llm_acc = accuracy_score(true_labels, llm_preds)
llm_prec = precision_score(true_labels, llm_preds, average='macro', zero_division=0)
llm_f1 = f1_score(true_labels, llm_preds, average='macro', zero_division=0)
llm_cm = confusion_matrix(true_labels, llm_preds)
all_metrics.append((llm_acc, llm_prec, llm_f1))
all_cms.append(llm_cm)
print("\nLLM Results:")
print(f"Accuracy: {llm_acc:.2f}")
print(f"Precision: {llm_prec:.2f}")
print(f"F1: {llm_f1:.2f}")
print("Confusion Matrix:\n", llm_cm)

# RAC
rac_preds = [rac.predict(text) for text in test_texts]
all_preds.append(rac_preds)
rac_acc = accuracy_score(true_labels, rac_preds)
rac_prec = precision_score(true_labels, rac_preds, average='macro', zero_division=0)
rac_f1 = f1_score(true_labels, rac_preds, average='macro', zero_division=0)
rac_cm = confusion_matrix(true_labels, rac_preds)
all_metrics.append((rac_acc, rac_prec, rac_f1))
all_cms.append(rac_cm)
print("\nRAC Results:")
print(f"Accuracy: {rac_acc:.2f}")
print(f"Precision: {rac_prec:.2f}")
print(f"F1: {rac_f1:.2f}")
print("Confusion Matrix:\n", rac_cm)

# --- Visualization ---
for i, (cm, method) in enumerate(zip(all_cms, methods)):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {method}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()



