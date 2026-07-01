"""
One-time migration script: re-embeds your existing ChromaDB collection
using a multilingual embedding model instead of all-MiniLM-L6-v2.

WHY THIS IS NEEDED:
all-MiniLM-L6-v2 is an English-only sentence transformer. Bangla queries
("বেলভি স্বাস্থ্য সুরক্ষা") don't reliably match against your mixed
Bangla/English documents in vector space — the model was never trained
on Bangla text, so similarity scores between a Bangla query and the
relevant chunk are weak and inconsistent. This is why retrieval was
incomplete even though the document existed in your collection.

WHAT THIS SCRIPT DOES:
1. Reads every document + metadata out of your existing 'bellevie_knowledge'
   collection (using the OLD embedder, just to extract the raw text — we
   are not re-using the old embeddings themselves)
2. Re-embeds every document using intfloat/multilingual-e5-base
3. Writes everything into a NEW collection 'bellevie_knowledge_multilingual'
   so your old collection is untouched and you can roll back instantly
   if anything looks wrong
4. Verifies the new collection with a real Bangla test query

IMPORTANT — e5 models require a "query: " / "passage: " prefix on every
input text. This is not optional styling — it's how the model was
trained, and skipping it measurably hurts retrieval quality. This script
and the corresponding main.py changes both apply this prefix correctly.

USAGE:
    python migrate_to_multilingual_embeddings.py

Run this from your project root (same place as main.py), with your
existing .venv activated, so it can find ./chroma_db.
"""

import os
import sys
import time
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY']     = 'False'

CHROMA_DIR           = os.getenv('CHROMA_DIR', './chroma_db')
OLD_COLLECTION_NAME  = 'bellevie_knowledge'
NEW_COLLECTION_NAME  = 'bellevie_knowledge_multilingual'
NEW_EMBEDDING_MODEL  = 'intfloat/multilingual-e5-base'

# e5 models need this prefix on every passage at embed time.
# (Queries at search time need "query: " instead — handled in main.py)
PASSAGE_PREFIX = "passage: "


def main():
    print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # ── Step 1: Read everything out of the old collection ──────
    try:
        old_collection = client.get_collection(OLD_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Could not open old collection '{OLD_COLLECTION_NAME}': {e}")
        print("   Check that CHROMA_DIR points at the right directory.")
        sys.exit(1)

    old_count = old_collection.count()
    print(f"✅ Found old collection with {old_count} vectors")

    if old_count == 0:
        print("❌ Old collection is empty — nothing to migrate.")
        sys.exit(1)

    print("Reading all documents, metadatas, and ids from old collection...")
    everything = old_collection.get(include=["documents", "metadatas"])

    ids       = everything["ids"]
    documents = everything["documents"]
    metadatas = everything["metadatas"] or [{} for _ in ids]

    print(f"✅ Pulled {len(documents)} documents out of old collection")

    # ── Step 2: Load the new multilingual embedding model ──────
    print(f"\nLoading new embedding model: {NEW_EMBEDDING_MODEL}")
    print("(first run downloads ~1.1GB — this is normal and only happens once)")
    t0 = time.time()
    new_embedder = SentenceTransformer(NEW_EMBEDDING_MODEL)
    print(f"✅ Model loaded in {time.time() - t0:.1f}s")

    # ── Step 3: Re-embed every document with the e5 passage prefix ──
    print(f"\nRe-embedding {len(documents)} documents...")
    print("(applying 'passage: ' prefix as required by the e5 model family)")

    prefixed_documents = [PASSAGE_PREFIX + doc for doc in documents]

    t0 = time.time()
    new_embeddings = new_embedder.encode(
        prefixed_documents,
        show_progress_bar=True,
        batch_size=16,
    ).tolist()
    print(f"✅ Re-embedded {len(new_embeddings)} documents in {time.time() - t0:.1f}s")

    # ── Step 4: Write into a NEW collection (old one untouched) ──
    print(f"\nCreating new collection: {NEW_COLLECTION_NAME}")

    # If a previous migration run left a partial collection, clean it up
    try:
        client.delete_collection(NEW_COLLECTION_NAME)
        print("(removed a previous partial migration of this collection)")
    except Exception:
        pass

    new_collection = client.create_collection(NEW_COLLECTION_NAME)

    # ChromaDB add() has a batch size ceiling on some backends — chunk it
    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids        = ids[i:i + BATCH_SIZE]
        batch_documents  = documents[i:i + BATCH_SIZE]          # store WITHOUT prefix
        batch_metadatas  = metadatas[i:i + BATCH_SIZE]
        batch_embeddings = new_embeddings[i:i + BATCH_SIZE]

        new_collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )
        print(f"  Inserted batch {i // BATCH_SIZE + 1} ({len(batch_ids)} docs)")

    new_count = new_collection.count()
    print(f"✅ New collection '{NEW_COLLECTION_NAME}' has {new_count} vectors")

    if new_count != old_count:
        print(f"⚠️  WARNING: old count ({old_count}) != new count ({new_count})")
        print("   Something may have been dropped — review before switching over.")
    else:
        print("✅ Document count matches — migration looks clean")

    # ── Step 5: Sanity-check with real Bangla queries ──────────
    print("\n" + "=" * 60)
    print("VERIFICATION — running real Bangla test queries")
    print("=" * 60)
    print("(using n_results=5 to match TOP_K_RETRIEVAL in main.py)\n")

    test_queries = [
        "বেলভি স্বাস্থ্য সুরক্ষা",                    # the exact query that was failing
        "BelleVie Health Protection System",          # English equivalent, for comparison
        "হৃদরোগ বিশেষজ্ঞ",                            # cardiologist
        "ভারতে চিকিৎসা",                              # treatment in India
    ]

    for query in test_queries:
        query_vector = new_embedder.encode("query: " + query).tolist()
        results = new_collection.query(
            query_embeddings=[query_vector],
            n_results=5,   # matches TOP_K_RETRIEVAL=5 in main.py
        )
        print(f"\nQuery: '{query}'")
        docs_returned = results["documents"][0]
        distances     = results.get("distances", [[None] * len(docs_returned)])[0]
        for rank, (doc, dist) in enumerate(zip(docs_returned, distances), 1):
            preview = doc[:100].replace("\n", " ")
            dist_str = f"{dist:.4f}" if dist is not None else "n/a"
            print(f"  [{rank}] (distance={dist_str}) {preview}...")

    # ── Step 6: Dedicated NGO coverage check ────────────────────
    # The NGO Health Protection System content is split across 6
    # chunks (NGO-001 through NGO-006: overview, 12 protection layers,
    # 20 platform features, 11 benefits, investment framework,
    # rationale). TOP_K_RETRIEVAL=5 in main.py means at most 5 of
    # these 6 can come back for any single query — this check shows
    # exactly how many do, and which one (if any) gets left out, so
    # you know what to expect before it shows up as a user complaint.
    print("\n" + "=" * 60)
    print("NGO COVERAGE CHECK — how many of the 6 NGO chunks come back")
    print("for a broad query, given TOP_K_RETRIEVAL=5 in main.py")
    print("=" * 60)

    ngo_query = "বেলভি স্বাস্থ্য সুরক্ষা ব্যবস্থা সম্পর্কে বিস্তারিত বলুন"  # "tell me in detail about..."
    query_vector = new_embedder.encode("query: " + ngo_query).tolist()
    results = new_collection.query(
        query_embeddings=[query_vector],
        n_results=5,
    )

    returned_ids = results["ids"][0]
    ngo_chunk_ids = {"NGO-001", "NGO-002", "NGO-003", "NGO-004", "NGO-005", "NGO-006"}
    ngo_returned  = [rid for rid in returned_ids if rid in ngo_chunk_ids]
    ngo_missing   = ngo_chunk_ids - set(ngo_returned)

    print(f"\nQuery: '{ngo_query}'")
    print(f"Top 5 results returned: {returned_ids}")
    print(f"NGO chunks included:    {sorted(ngo_returned)} ({len(ngo_returned)}/6)")
    if ngo_missing:
        print(f"NGO chunks NOT in top 5: {sorted(ngo_missing)}")
        print("⚠️  A broad question about the Health Protection System will likely")
        print("   miss this content. Consider whether the missing chunk(s) cover")
        print("   something users frequently ask about specifically.")
    else:
        print("✅ All 6 NGO chunks retrieved — full content will reach the LLM")

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"""
Next steps:
1. Review the test query results above — confirm the BelleVie Health
   Protection System content is showing up clearly for the Bangla query
2. Check the NGO COVERAGE CHECK section above — if chunks are missing,
   decide whether that matters enough to revisit chunk size/count later
3. Update main.py (see the updated version provided alongside this script):
     EMBEDDING_MODEL = '{NEW_EMBEDDING_MODEL}'
     collection = chroma_client.get_collection('{NEW_COLLECTION_NAME}')
     TOP_K_RETRIEVAL = 5
   and apply the 'query: ' prefix in retrieve()
4. Restart and test against real queries
5. Once confirmed working in production, you can optionally delete the
   old collection to save disk space:
     client.delete_collection('{OLD_COLLECTION_NAME}')
   (not done automatically by this script — your call, once you're confident)
""")


if __name__ == "__main__":
    main()
