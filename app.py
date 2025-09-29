import streamlit as st
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import re
import sqlite3
import time

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # embedding size for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ========================
# PDF Extraction Helpers
# ========================
def extract_text_from_pdf(file):
    text_per_page = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_per_page.append(text)
    return text_per_page

def chunk_text_safe(text_input, max_words=1000):
    chunks = []
    if isinstance(text_input, list):
        for text in text_input:
            words = str(text).split()
            chunks.extend([" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)])
    else:
        words = str(text_input).split()
        chunks.extend([" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)])
    return chunks




# ========================
# Streamlit Menu
# ========================
menu = ["Upload CSV & Create Vector DB", "Upload PDF & Create Vector DB", "Upload DOCX & Create Vector DB", "Chatbot"]
choice = st.sidebar.selectbox("Menu", menu)




# ========================
# CSV Upload Section
# ========================
import tiktoken

if choice == "Upload CSV & Create Vector DB":
    st.header("📂 Upload CSV and Store in Pinecone")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    namespace = st.text_input("Enter a unique namespace for Pinecone storage:")

    if uploaded_file is not None and namespace:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV Loaded Successfully!")

        # Fill empty cells according to column type
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(0, inplace=True)
            elif pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                df[col].replace(["", "nan", "None", "NaN"], "0", inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col].fillna("0", inplace=True)

        df.reset_index(drop=True, inplace=True)
        st.dataframe(df.head())

        # Store data in SQLite
        chat_db_name = "chat_data.db"
        conn = sqlite3.connect(chat_db_name)
        df.to_sql(namespace, conn, if_exists="replace", index=False)
        conn.close()
        st.info(f"📦 Data stored in SQLite DB: {chat_db_name}, table: '{namespace}'")

        # Convert row into sentence
        def row_to_sentence_direct(row: pd.Series) -> str:
            parts = []
            for col, val in row.items():
                if pd.notna(val) and str(val).strip() != "0":
                    parts.append(f"{col}: {val}")
            return ". ".join(parts)

        # Token-based chunking to avoid max token limit
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        def chunk_text_by_tokens(text_input, max_tokens=8000):
            chunks = []
            texts = text_input if isinstance(text_input, list) else [text_input]

            for text in texts:
                words = str(text).split()
                current_chunk = []
                current_tokens = 0

                for word in words:
                    word_tokens = len(enc.encode(word))
                    if current_tokens + word_tokens > max_tokens:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_tokens = word_tokens
                    else:
                        current_chunk.append(word)
                        current_tokens += word_tokens

                if current_chunk:
                    chunks.append(" ".join(current_chunk))
            return chunks

        # Embed row-wise with token-based chunking and connected metadata
        status_text = st.empty()
        progress_bar = st.progress(0)

        for i, row in df.iterrows():
            sentence = row_to_sentence_direct(row)
            chunks = chunk_text_by_tokens(sentence, max_tokens=8000)

            for j, chunk in enumerate(chunks):
                embedding = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                ).data[0].embedding

                vector = {
                    "id": f"{namespace}-row-{i}-chunk-{j}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "row_id": i,        # All chunks share same row_id
                        "chunk_id": j,      # Individual chunk index
                        "total_chunks": len(chunks)  # Total chunks for this row
                    }
                }
                index.upsert(vectors=[vector], namespace=namespace)

            status_text.info(f"Processed {i+1}/{len(df)} rows")
            progress_bar.progress((i+1)/len(df))

        st.success(f"✅ CSV rows embedded in Pinecone with connected chunks under namespace '{namespace}'.")


# ========================
# pdf Section
# ========================
elif choice == "Upload PDF & Create Vector DB":
    st.header("📄 Upload PDF & Create Vector DB")

    client_namespace = st.text_input("Enter a unique client namespace (required)")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file and client_namespace:
        uploaded_file.seek(0)

        import pdfplumber
        all_text_rows = []
        all_tables_rows = []
        combined_columns = []

        # -----------------------------
        # 🔹 Extract text and tables
        # -----------------------------
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                # Plain text
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text_rows.append(page_text.strip())

                # Tables
                tables = page.extract_tables()
                for table in tables:
                    if len(table) > 1:  # header + rows
                        header = table[0]
                        combined_columns.extend(header)
                        all_tables_rows.extend(table[1:])

        st.write("📖 Preview of text:")
        st.write(all_text_rows[0][:1000] + "..." if all_text_rows else "No text found.")

        # -----------------------------
        # 🔹 Clean & deduplicate column names
        # -----------------------------
        def clean_and_dedup_columns(columns):
            seen = {}
            cleaned = []
            for i, col in enumerate(columns):
                if not col or str(col).strip() == "":
                    col = f"col_{i}"
                else:
                    col = str(col).strip().replace("\n", "_").replace(" ", "_")
                    col = "".join(c if c.isalnum() or c == "_" else "_" for c in col)
                if col in seen:
                    seen[col] += 1
                    col = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
                cleaned.append(col)
            return cleaned

        safe_columns = clean_and_dedup_columns(list(dict.fromkeys(combined_columns)))

        # -----------------------------
        # 🔹 Convert tables to text rows
        # -----------------------------
        table_text_rows = []
        for row in all_tables_rows:
            # Match length of columns
            if len(row) < len(safe_columns):
                row.extend([""] * (len(safe_columns) - len(row)))
            elif len(row) > len(safe_columns):
                row = row[:len(safe_columns)]
            row_text = ", ".join([str(r) for r in row])
            table_text_rows.append(row_text)

        # -----------------------------
        # 🔹 Combine table rows + plain text
        # -----------------------------
        combined_rows = table_text_rows + all_text_rows
        df_combined = pd.DataFrame({"text": combined_rows})

        # -----------------------------
        # 🔹 Store everything in a single SQLite table
        # -----------------------------
        conn = sqlite3.connect("chat_data.db")
        df_combined.to_sql(client_namespace, conn, if_exists="replace", index=False)
        conn.close()

        st.success(f"✅ PDF fully processed and stored in SQLite as single table '{client_namespace}'")

        # -----------------------------
        # 🔹 Prepare Pinecone embeddings (safe)
        # -----------------------------
        EMBED_MODEL = "text-embedding-3-small"
        sentences = combined_rows  # all table rows + text

        chunks = chunk_text_safe(sentences)
        st.info(f"📦 Split into {len(chunks)} chunks for Pinecone embeddings")

        status_text = st.empty()
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks, start=1):
            embedding = client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding
            index.upsert(
                vectors=[{
                    "id": f"{client_namespace}-chunk-{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "chunk_id": i, "total_chunks": len(chunks)}
                }],
                namespace=client_namespace
            )
            status_text.info(f"Processing chunk {i}/{len(chunks)}")
            progress_bar.progress(i / len(chunks))

        st.success(f"✅ PDF fully processed and stored in Pinecone under namespace '{client_namespace}'")

# ========================
# dox Section
# ========================
# ========================
# DOCX Section
# ========================
elif choice == "Upload DOCX & Create Vector DB":
    st.title("📄 Upload DOCX & Create Vector DB")

    client_namespace = st.text_input("Enter a unique client namespace (required)")
    uploaded_file = st.file_uploader("Upload DOCX file", type=["docx"])

    def extract_text_from_docx(file):
        from docx import Document
        doc = Document(file)
        return [para.text.strip() for para in doc.paragraphs if para.text.strip() != ""]

    if uploaded_file and client_namespace:
        # -----------------------------
        # 🔹 Extract DOCX text
        # -----------------------------
        all_text_rows = extract_text_from_docx(uploaded_file)
        st.write("📖 Preview of extracted text:")
        st.write(all_text_rows[0][:1000] + "..." if all_text_rows else "No text found.")

        # -----------------------------
        # 🔹 1️⃣ Store in SQLite (row-wise dataset)
        # -----------------------------
        import sqlite3
        df_docx = pd.DataFrame({"text": all_text_rows})
        conn = sqlite3.connect("chat_data.db")
        df_docx.to_sql(client_namespace, conn, if_exists="replace", index=False)
        conn.close()
        st.success(f"✅ DOCX stored in SQLite table '{client_namespace}' (row-wise)")

        # -----------------------------
        # 🔹 2️⃣ Prepare Pinecone embeddings (full-text dataset, chunked if too long)
        # -----------------------------
        if st.button("Generate Embeddings for Pinecone"):
            EMBED_MODEL = "text-embedding-3-small"

            # Combine all rows into a single string
            full_text = " ".join(all_text_rows)
            chunks = chunk_text_safe(full_text)  # split long text into manageable chunks

            st.info(f"📦 Split full text into {len(chunks)} chunks for Pinecone embeddings")
            status_text = st.empty()
            progress_bar = st.progress(0)

            for i, chunk in enumerate(chunks, start=1):
                embedding = client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding
                index.upsert(
                    vectors=[{
                        "id": f"{client_namespace}-chunk-{i}",
                        "values": embedding,
                        "metadata": {"text": chunk, "chunk_id": i, "total_chunks": len(chunks)}
                    }],
                    namespace=client_namespace
                )
                status_text.info(f"Processing chunk {i}/{len(chunks)}")
                progress_bar.progress(i / len(chunks))

            st.success(f"✅ Full DOCX text stored in Pinecone under namespace '{client_namespace}'")
    else:
        st.info("Please upload a DOCX file and enter a namespace.")



# ========================
# Chatbot Section
elif choice == "Chatbot":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Chat with your Knowledge Base</h1>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_namespace = st.text_input("🔑 Enter namespace to chat with", key="namespace")

    if chat_namespace:
        db_name = "chat_data.db"
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 🔎 Check if namespace exists in SQLite
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (chat_namespace,))
        table_exists = cursor.fetchone()

        if not table_exists:
            st.error(f"❌ No table found for namespace '{chat_namespace}'. Please upload CSV first.")
        else:
            # ✅ Load table data
            df = pd.read_sql(f"SELECT * FROM '{chat_namespace}'", conn)
            conn.close()

            # 🗨️ Show chat history
            for role, content in st.session_state.chat_history:
                avatar = "👤" if role == "user" else "🤖"
                with st.chat_message(role, avatar=avatar):
                    if isinstance(content, dict):  # assistant with resources
                        st.markdown(content["reply"])
                        st.caption(f"✅ Mode used: **{content['mode'].upper()}** (reason: {content['reason']})")

                        with st.expander("📖 Resources used", expanded=False):
                            st.markdown("**Analytical Output:**")
                            st.markdown(content["analytical"] if content["analytical"] else "_No analytical result_")

                            st.markdown("---")
                            st.markdown("**Semantic Evidence (Top Rows):**")
                            if content["semantic"]:
                                for score, row_id, row_text in content["semantic"][:5]:
                                    st.markdown(f"- **Row {row_id}** (score {round(score,3)}): {row_text}")
                            else:
                                st.markdown("_No semantic rows found_")
                    else:
                        st.markdown(content)

            # 🔎 User input
            user_input = st.chat_input("Ask something...")

            if user_input:
                st.session_state.chat_history.append(("user", user_input))
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)

                with st.spinner("Thinking..."):

                    # ======================================================
                    # 1️⃣ DECIDE MODE
                    # ======================================================
                    mode_prompt = f"""
The user asked: "{user_input}".

You have two possible modes:
1. Analytical → Run pandas code on DataFrame (numerical, filtering, grouping, statistics).
2. Semantic → Search dataset text rows for insights (descriptive, qualitative, general questions).

Task:
- Decide which mode is more suitable.
- Reply ONLY in JSON:

{{
  "mode": "analytical" OR "semantic",
  "reason": "short reason"
}}
"""
                    mode_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": mode_prompt}],
                        temperature=0
                    )

                    import json
                    try:
                        decision = json.loads(mode_resp.choices[0].message.content.strip())
                        chosen_mode = decision.get("mode", "semantic")
                        reason = decision.get("reason", "No reason provided.")
                    except:
                        chosen_mode = "semantic"
                        reason = "Fallback to semantic due to JSON parse error."

                    final_reply = ""
                    analytical_answer, semantic_answer = None, None
                    re_ranked = []

                    # ======================================================
                    # 2️⃣ ANALYTICAL MODE
                    # ======================================================
                    if chosen_mode == "analytical":
                        try:
                            import re
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="ignore")
                                if df[col].dtype == "object":
                                    try:
                                        df[col] = pd.to_datetime(df[col], errors="raise")
                                    except:
                                        pass

                            prompt_code = f"""
You are a Python data assistant.
The DataFrame df has columns: {list(df.columns)}.

User question: "{user_input}"

Task:
- Write ONE line of pandas code using df to answer the question.
- Return only Python code inside ```python ... ```
"""
                            resp_code = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt_code}],
                                temperature=0
                            )
                            raw_code_reply = resp_code.choices[0].message.content.strip()

                            code_match = re.search(r"```python\n(.*?)```", raw_code_reply, re.DOTALL)
                            code = code_match.group(1).strip() if code_match else raw_code_reply.strip()

                            result = eval(code, {"df": df, "pd": pd})
                            if isinstance(result, pd.DataFrame):
                                result_text = result.to_markdown()
                            elif isinstance(result, pd.Series):
                                result_text = result.to_string()
                            else:
                                result_text = str(result)

                            prompt_summary = f"""
The user asked: "{user_input}".
Here is the pandas result:

{result_text}

Summarize in clear English.
"""
                            resp_summary = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt_summary}],
                                temperature=0.2
                            )
                            analytical_answer = resp_summary.choices[0].message.content.strip()
                            final_reply = analytical_answer
                        except Exception as e:
                            final_reply = f"⚠️ Analytical mode failed: {e}"

                    # ======================================================
                    # 3️⃣ SEMANTIC MODE
                    # ======================================================
                    else:
                        try:
                            emb_resp = client.embeddings.create(
                                model="text-embedding-3-small",
                                input=user_input
                            )
                            query_emb = emb_resp.data[0].embedding

                            result = index.query(
                                vector=query_emb,
                                top_k=10,
                                namespace=chat_namespace,
                                include_metadata=True
                            )
                            matches = result.get("matches", [])

                            keywords = [w.lower() for w in user_input.split()]
                            for match in matches:
                                meta = match.get("metadata", {})
                                score = match.get("score", 0)
                                row_text = meta.get("text", "").strip()
                                row_id = meta.get("row_id", meta.get("chunk_id", "N/A"))
                                keyword_hits = sum(1 for k in keywords if k in row_text.lower())
                                final_score = score + (0.05 * keyword_hits)
                                re_ranked.append((final_score, row_id, row_text))

                            re_ranked = sorted(re_ranked, key=lambda x: x[0], reverse=True)[:15]

                            batch_size = 5
                            batch_summaries = []
                            for i in range(0, len(re_ranked), batch_size):
                                batch = re_ranked[i:i+batch_size]
                                batch_context = "\n".join([f"- {t[2]}" for t in batch])

                                prompt = f"""
Analyze these rows for query: "{user_input}"

{batch_context}

Instructions:
- Only use given rows, no invention.
- Summarize relevant insights.
"""
                                resp = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.3
                                )
                                batch_summaries.append(resp.choices[0].message.content.strip())

                            merged_context = "\n".join(batch_summaries)
                            final_prompt = f"""
User Question: {user_input}

Insights from dataset rows:
{merged_context}

Give a clear, structured final answer (no invention).
"""
                            final_resp = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": final_prompt}],
                                temperature=0.2
                            )
                            semantic_answer = final_resp.choices[0].message.content.strip()
                            final_reply = semantic_answer
                        except Exception as e:
                            final_reply = f"⚠️ Semantic mode failed: {e}"

                    # ======================================================
                    # DISPLAY + SAVE
                    # ======================================================
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(final_reply)
                        st.caption(f"✅ Mode used: **{chosen_mode.upper()}** (reason: {reason})")

                        with st.expander("📖 Resources used", expanded=False):
                            st.markdown("**Analytical Output:**")
                            st.markdown(analytical_answer if analytical_answer else "_No analytical result_")

                            st.markdown("---")
                            st.markdown("**Semantic Evidence (Top Rows):**")
                            if re_ranked:
                                for score, row_id, row_text in re_ranked[:5]:
                                    st.markdown(f"- **Row {row_id}** (score {round(score,3)}): {row_text}")
                            else:
                                st.markdown("_No semantic rows found_")

                    # ✅ Save full dict (reply + resources)
                    st.session_state.chat_history.append((
                        "assistant",
                        {
                            "reply": final_reply,
                            "mode": chosen_mode,
                            "reason": reason,
                            "analytical": analytical_answer,
                            "semantic": re_ranked
                        }
                    ))
