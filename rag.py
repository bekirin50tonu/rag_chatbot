# -*- coding: utf-8 -*-
"""RAG

# Akbank GenAI Bootcamp: RAG Chatbot Projesi

## Projenin Amacı

Bu proje, RAG (Retrieval Augmented Generation) mimarisi kullanarak **Metin/WikiRAG-TR** veri seti üzerinde çalışan bir **Soru-Cevap (Q&A)** chatbot geliştirmeyi amaçlamaktadır. Proje, **Gemini API**'ı kullanarak veriye dayalı, doğru ve bağlamsal olarak zenginleştirilmiş yanıtlar sunmayı hedeflemektedir.

**Kullanılan Veri Seti:**
* https://huggingface.co/datasets/Metin/WikiRAG-TR


**Kullanılan Yöntemler ve Teknolojiler:**

* Generation Model: Gemini API
    * Google'ın geliştirdiği model olup, Gemini API ve **google-genai SDK** ile hızlıca uygulanabilmesinin, Türkçe RAG görevlerinde kullanım kolaylığı sağlamaktadır. Kullanımı için API anahtarı gerekmektedir.

* RAG Framework: LangChain
    * RAG projeleri için en yaygın kullanılan ve en geniş topluluk desteğine sahip framework'tür. Gemini API'a özgü entegrasyonları hızlıca uygulanabilir.

* Embedding Model: trmteb/turkish-embedding-model
    * Gemini API ile sorunsuz entegrasyon sağlar ve genellikle LangChain ile uyumludur. Hugging Face platformunda yer alan bu model için API anahtarının bulunması gerekmektedir.

* Vektör Veritabanı: Chroma
    * Kurulumu ve kullanımı son derece kolay, hafif (in-memory) bir veritabanıdır. Colab ortamında hızlı prototipleme ve test için idealdir.

* Web Arayüzü: Gradio
    * Python kodu üzerinden minimal çabayla profesyonel ve etkileşimli web uygulamaları oluşturmanızı sağlayan, projenin arayüz katmanı, Makine Öğrenimi demoları için optimize edilmiş ve Colab Notebook ortamında son derece stabil çalışan kütüphanedir.

## Detaylı RAG Test Sonuçları

**Sonuçlar Özeti**:

* Veriseti, 3-5 Saniyede Yüklendi.
* Toplam Öğe Sayısı 5000.
* Parçalanmış belge sayısı: 20324.
* Vektör aşamasında 2GB sistem, 1.8GB GPU RAM ihtiyaç duymuştur.
* Pipeline, 300-314 Saniyede Yüklendi.
* Zincir, 4 Saniyede Yüklendi:
"""

# =========================================================================
# 1. Ortam Değişkenlerini Kontrol Etme
# =========================================================================
import os
from dotenv import load_dotenv

# .env dosyasını yükler. Aynı dizindeki dosyaları arar.
load_dotenv()


# Google/Gemini Anahtarını Yükleme
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    print(
        "✅ Google API Anahtarı (.env veya Ortam Değişkenleri) üzerinden yüklendi.",
        os.environ["GOOGLE_API_KEY"],
    )
else:
    print("❌ HATA: 'GOOGLE_API_KEY' ortam değişkeni bulunamadı.")
    print(
        "Lütfen .env dosyanızı kontrol edin ve anahtarınızı doğru girdiğinizden emin olun."
    )
    # Anahtar olmadan kodun çalışmasını durdurmak için:
    raise EnvironmentError("GOOGLE_API_KEY olmadan devam edilemiyor.")


# Hugging Face Token'ı Yükleme (Gerekliyse)
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    print(
        "✅ Hugging Face Token (.env veya Ortam Değişkenleri) üzerinden yüklendi.",
        os.environ["HF_TOKEN"],
    )
else:
    print("⚠️ Uyarı: HF_TOKEN bulunamadı. Yerel Embedding modelinizde sorun çıkabilir.")

# Device Bilgisi Yükleme (Eğer yoksa cpu olacak)
if os.getenv("DEVICE"):
    os.environ["DEVICE"] = os.getenv("DEVICE")
    print(
        "✅ DEVICE bilgisi (.env veya Ortam Değişkenleri) üzerinden yüklendi.",
        os.environ["DEVICE"],
    )
else:
    print("⚠️ Uyarı: DEVICE bulunamadı. DEVICE, cpu olarak seçilecektir.")
    os.environ["DEVICE"] = "cpu"

# =========================================================================
# 2. Veri Setini İndirme ve Detayları Görüntüleme
# =========================================================================

from datasets import load_dataset
from langchain_core.documents import Document
from typing import List


def load_dataset_and_preview(dataset_name: str, select: int = 5000) -> List[Document]:
    """
    Belirtilen Hugging Face veri setini yükler, ilk 'select' kadarını alır ve
    LangChain Document objeleri listesine dönüştürür.
    İstenen sütunlar metadata olarak saklanır.

    Args:
        dataset_name (str): Yüklenecek Hugging Face veri setinin adı (Örn: "Metin/WikiRAG-TR").
        select (int): Yüklenecek belge sayısı. Prototip için varsayılan 5000'dir.

    Returns:
        List[Document]: LangChain formatına dönüştürülmüş belgelerin listesi.
    """

    # Hugging Face'den veri setini yükleme
    print("Veri seti yükleniyor...")
    # split="train" ile tren setini yüklüyor ve ilk 'select' kadarını alıyoruz.
    dataset = load_dataset(
        dataset_name, split="train", token=os.getenv("HF_TOKEN")
    ).select(range(select))
    print("Veri seti başarıyla yüklendi.")

    # Toplam Öğe Sayısı
    N_DOCUMENTS = len(dataset)
    print(f"Toplam Öğe Sayısı: {N_DOCUMENTS}")

    all_documents = []

    # Metadata olarak saklanacak sütunlar
    METADATA_COLS = ["id", "question", "answer"]

    for item in dataset:
        # Kaynak metin için 'context' sütununu kullanıyoruz
        page_content = item.get(
            "context", "Context Not Found"
        )  # Güvenlik için get() kullandık

        # Metadata objesini oluşturma
        metadata = {col: item.get(col) for col in METADATA_COLS}
        metadata["source"] = (
            dataset_name  # Veri seti adını dinamik olarak kaynağa yazıyoruz
        )

        # LangChain Document objesini oluşturma
        doc = Document(page_content=page_content, metadata=metadata)
        all_documents.append(doc)

    print(
        f"\nLangChain formatına dönüştürülen toplam belge sayısı: {len(all_documents)}"
    )

    # İlk 3 Metin Parçasının LangChain Document Formatında Önizlemesi
    print("\n--- İlk 3 Metin Parçasının LangChain Document Formatında Önizlemesi ---")
    for doc in all_documents[:3]:
        print(f"ID: {doc.metadata.get('id', 'Yok')}")
        print(f"Question (Metadata): {doc.metadata.get('question', 'Yok')}")
        print(f"Answer (Metadata): {doc.metadata.get('answer', 'Yok')[:100]}...")
        print(f"İçerik (Context - İlk 100 Karakter): {doc.page_content[:100]}...")
        print("-" * 30)

    return all_documents


# =========================================================================
# 3. RAG Pipeline Oluşturma.
# =========================================================================

from typing import List, Tuple

# LangChain sınıflarının içe aktarılması
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
import torch


def setup_rag_pipeline(
    all_documents: List[Document], model_name: str = "trmteb/turkish-embedding-model"
) -> Tuple[HuggingFaceEmbeddings, Chroma, BaseRetriever]:
    """
    RAG pipeline'ı için embedding modelini tanımlar, belgeleri vektörleştirir
    ve ChromaDB'ye kaydeder, ardından bir retriever döndürür.

    Döndürülen değerler (Tuple):
    - GoogleGenAIEmbeddings: Kullanılan embedding modelinin nesnesi.
    - Chroma: Vektör veritabanının nesnesi.
    - BaseRetriever: Arama yapabilen retriever nesnesi.
    """
    print("Vektör veritabanı ve RAG zinciri kuruluyor...")

    # BELGE PARÇALAMA (CHUNKING)
    print("Belgeler parçalanıyor (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        # Ayırıcı listesi belirtilmediğinde varsayılan akıllı listeyi kullanır
    )

    # Parçalanmış belgeleri oluşturma
    chunked_documents = text_splitter.split_documents(all_documents)
    print(
        f"Orijinal belge sayısı: {len(all_documents)}. Parçalanmış belge sayısı: {len(chunked_documents)}"
    )

    # 1. Google Embedding Modelini Tanımlama
    print(f"Yerel Türkçe Embedding modeli yükleniyor: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": os.environ["DEVICE"]},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 512},
    )

    # 2. Vektör Veritabanı Oluşturma (Indexing)
    print(
        f"Toplam {len(all_documents)} belge vektörleştirilip ChromaDB'ye kaydediliyor. Lütfen bekleyiniz..."
    )

    vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        collection_name="wikirag-tr-rag",
    )

    print("\nVektörleştirme ve Indexing (Dizinleme) işlemi tamamlandı!")
    print(f"Chroma Veritabanı (Collection: 'wikirag-tr-rag') hazırlandı.")

    # 3. Retriever (Geri Çağırıcı) Tanımlama
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    print("Retriever (Geri Çağırıcı) LangChain için tanımlandı.")

    # İstenen 3 nesneyi Tuple olarak döndürüyoruz
    return embeddings, vectorstore, retriever


# =========================================================================
# 4. Modeli ve Zinciri Oluşturma
# =========================================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable


def setup_model_and_chain(
    retriever: BaseRetriever, model_name: str = "gemini-2.5-flash"
) -> Runnable:
    """
    Gemini LLM, Prompt Template ve Retriever kullanarak LangChain RAG zincirini (Runnable) oluşturur.

    Args:
        retriever (BaseRetriever): ChromaDB'den gelen ve arama yapabilen Retriever nesnesi.
        model_name (str): Kullanılacak Gemini modelinin adı.

    Returns:
        Runnable: Tüm RAG akışını içeren LangChain zinciri (LCEL uyumlu ana arayüz).
    """

    # 1. LLM (Generative Model) Tanımlama
    print(f"{model_name} modeli tanımlanıyor...")
    llm = ChatGoogleGenerativeAI(
        model=model_name, temperature=0.0, request_options={"timeout": 60}
    )

    # 2. Prompt Şablonu Oluşturma
    prompt_template = """Sen bir Türkçe RAG (Bilgiye Dayalı Soru-Cevap) asistanısın.
Aşağıdaki 'context' (bağlam) içinde yer alan bilgileri kullanarak kullanıcının sorusunu yanıtla.
Eğer bağlamda soruya net bir cevap bulamıyorsan, kibarca 'Elimdeki bilgilere göre bu soruya net bir yanıt veremiyorum.' şeklinde cevap ver.
Asla verilen bağlam dışına çıkma ve kendi genel bilginle cevap verme.

Context:
---
{context}
---

Soru: {input}
Yanıt:
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. Dokümanları Birleştirme Zinciri (Document Combination Chain) Oluşturma
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 4. Ana RAG Zincirini (Chain) Oluşturma
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG Pipeline (Zinciri) başarıyla oluşturuldu!")

    # 5. Pipeline Testi
    print("\n--- Pipeline Test Ediliyor ---")
    test_queries = [
        "1923'te Türkiye'de hangi önemli olay gerçekleşmiştir?",
        "Tengri sözcüğünün anlamı nedir?",
        "Alan Tuning kimdir?",
    ]
    for test_query in test_queries:
        response = rag_chain.invoke({"input": test_query})

        print(f"\nSoru: {test_query}")
        print(f"\nGemini Yanıtı:\n{response['answer']}")
        print("\n--- Kullanılan Kaynaklar (Context) ---\n")
        for i, doc in enumerate(response["context"]):
            print(
                f"Kaynak {i+1} (ID: {doc.metadata.get('id', 'N/A')} - Soru: {doc.metadata.get('question', 'Yok')[:40]}...):"
            )
            # Kaynak içeriğinin ilk 500 karakterini yazdırır
            print(f"İçerik Başlangıcı: {doc.page_content.strip()[:500]}...")
            print("-" * 30)

    return rag_chain


# =========================================================================
# 5. Ortamı Hazırlama Ve Detayları Alma
# =========================================================================

from datetime import datetime

# Pipeline'ı yükle
try:
    # Veri Setini Yükleme ve Dokümanlara Dönüştürme
    start_time = datetime.now()
    print("\nVeriler Hazırlanıyor...")
    all_documents = load_dataset_and_preview(
        dataset_name="Metin/WikiRAG-TR", select=100
    )
    dataset_load_time = datetime.now() - start_time
    print(f"\nVeriseti, {dataset_load_time.seconds} Saniyede Yüklendi: ")
    print("=" * 30)

    # 2. Embedding ve Indexing
    start_time = datetime.now()
    print("\nRAG Pipeline Hazırlanıyor...")
    embeddings, vectorstore, retriever = setup_rag_pipeline(
        all_documents, model_name="trmteb/turkish-embedding-model"
    )
    pipeline_load_time = datetime.now() - start_time
    print(f"\nPipeline, {pipeline_load_time.seconds} Saniyede Yüklendi: ")
    print("=" * 30)

    # 3. LLM ve Chain Tanımlama
    start_time = datetime.now()
    print("\nModel ve Retriever Hazırlanıyor...")
    rag_chain = setup_model_and_chain(retriever, model_name="gemini-2.5-flash")
    setup_chain_time = datetime.now() - start_time
    print(f"\nZincir, {setup_chain_time.seconds} Saniyede Yüklendi: ")
    print("=" * 30)

except Exception as e:
    print(f"RAG Pipeline Kurulumunda Hata: {e}")
    exit()

# =========================================================================
# 6. Gradio İle Görselleştirme
# =========================================================================

import gradio as gr
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings


def chat_interface_handler(query: str, history: list) -> str:
    """
    Gradio'dan gelen yeni sorguyu ve sohbet geçmişini alır,
    RAG zincirini çalıştırır ve yanıtı döndürür.
    """

    # LangChain'in anlayacağı formata (HumanMessage/AIMessage) çevirme
    # Gradio history formatı: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    langchain_history = []
    for human, ai in history:
        langchain_history.append(HumanMessage(content=human))
        langchain_history.append(AIMessage(content=ai))

    # Güncel sorguyu ekleme
    langchain_history.append(HumanMessage(content=query))

    try:
        # RAG Zincirini çalıştırma
        # LangChain'in 'runnable' zincirleri, history'yi 'chat_history' anahtarıyla alabilir.
        response = rag_chain.invoke({"input": query, "chat_history": langchain_history})

        # Kaynakları ve yanıtı ayırma
        answer = response["answer"]
        context_docs = response["context"]

        # Kaynakları daha okunaklı bir formatta (HTML) yanıta ekleme
        sources_markdown = "🔍 Kaynaklar:\n"

        for i, doc in enumerate(context_docs):
            content = doc.page_content.replace("\n", " ").strip()
            doc_id = doc.metadata.get("id", "N/A")

            # Markdown içindeki details etiketi (Gradio'nun daha iyi işlediği format)
            sources_markdown += f"""
<details>
<summary>Kaynak {i+1} (ID: {doc_id} - Q: {doc.metadata.get('question', 'Yok')[:50]}...)</summary>
{content}
</details> """
        final_response_with_sources = f"{answer}<br><br>{sources_markdown}"

        return final_response_with_sources

    except Exception as e:
        return f"Sorgu sırasında hata oluştu: {e}"


# gr.ChatInterface bileşenini kullanma
chat_interface = gr.ChatInterface(
    fn=chat_interface_handler,
    textbox=gr.Textbox(
        placeholder="Sormak istediğiniz soruyu buraya yazın...",
        container=False,
        scale=7,
    ),
    title="🇹🇷 Gemini RAG Chatbot (Sohbet Modu)",
    description="Türkçe Yerel Embedding ve Gemini 2.5 Flash ile desteklenen, sohbet geçmişini takip eden RAG uygulaması.",
    theme="soft",
    examples=[
        ["Türkiye Cumhuriyeti'ni kim kurdu?"],
        ["Kurucunun siyasi hayatı hakkında bilgi ver."],
        ["Bu kişi hangi savaşlara katıldı?"],
    ],
)

# Gradio'yu başlatma
print("Gradio Sohbet Arayüzü başlatılıyor...")
chat_interface.launch(inline=True, share=False)
