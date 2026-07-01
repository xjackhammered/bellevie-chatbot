"""
Diagnostic script — checks what's actually stored in your ChromaDB
for the BelleVie Health Protection System / NGO document content.

This answers one question: is the text in your vector store clean,
or does it carry the same character-level corruption visible in the
source PDF (broken Bangla conjuncts, e.g. "এন্জজও" instead of "এনজিও")?

This matters because even a perfect multilingual embedding model
can't match a clean Bangla query against corrupted document text —
the character sequences won't line up. If corruption is confirmed
here, the fix is to re-extract/re-clean the source text BEFORE
running the embedding migration, not after.

USAGE:
    python check_ngo_document_in_chromadb.py

Run from your project root (same place as main.py), .venv activated.
Checks your CURRENT collection (whatever EMBEDDING_MODEL/COLLECTION_NAME
your main.py is presently configured with — works whether or not
you've run the multilingual migration yet).
"""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY']     = 'False'

CHROMA_DIR = os.getenv('CHROMA_DIR', './chroma_db')

# Try both possible collection names — whichever exists tells us
# where you currently are in the migration process.
CANDIDATE_COLLECTIONS = [
    'bellevie_knowledge_multilingual',  # post-migration name
    'bellevie_knowledge',               # original name
]

SEARCH_TERMS = ['BelleVie', 'NGO', 'এনজিও', 'স্বাস্থ্য', 'Health Protection']


def main():
    print(f"Connecting to ChromaDB at {CHROMA_DIR}...\n")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing_collections = [c.name for c in client.list_collections()]
    print(f"Collections found in this ChromaDB: {existing_collections}\n")

    for name in CANDIDATE_COLLECTIONS:
        if name not in existing_collections:
            continue

        print("=" * 70)
        print(f"Inspecting collection: {name}")
        print("=" * 70)

        collection = client.get_collection(name)
        total_count = collection.count()
        print(f"Total documents in collection: {total_count}\n")

        # Pull everything so we can search raw text directly —
        # this bypasses embeddings entirely, just checking stored text.
        everything = collection.get(include=["documents", "metadatas"])
        docs = everything["documents"]
        ids  = everything["ids"]

        # Find any document mentioning the NGO / Health Protection content
        matches = []
        for doc_id, doc in zip(ids, docs):
            if any(term in doc for term in ['BelleVie', 'NGO', 'এনজিও', 'এন্জজও', 'Health Protection']):
                matches.append((doc_id, doc))

        print(f"Documents matching NGO/Health Protection terms: {len(matches)}\n")

        if not matches:
            print("⚠️  NO matching documents found at all in this collection.")
            print("    This means the document either:")
            print("    (a) was never successfully embedded into this collection, or")
            print("    (b) the chunk text doesn't contain any of the expected terms\n")
            continue

        for i, (doc_id, doc) in enumerate(matches, 1):
            print(f"--- Match {i} (id: {doc_id}) ---")
            print(f"Length: {len(doc)} characters")
            preview = doc[:300].replace("\n", " ")
            print(f"Preview: {preview}")

            # Check for visible corruption markers — fragmented conjuncts
            # often show up as isolated halant/virama characters (্) followed
            # by inconsistent spacing, or zero-width joiners in odd places
            halant_count = doc.count('্')
            char_count   = len(doc.replace(' ', ''))
            halant_ratio = halant_count / max(char_count, 1)

            print(f"Halant (্) density: {halant_ratio:.1%} of characters")
            if halant_ratio > 0.08:
                print("⚠️  HIGH halant density — likely indicates fragmented/corrupted")
                print("    conjunct characters, similar to what's visible in the source PDF.")
            else:
                print("✅ Halant density looks normal for Bangla text.")
            print()

        print()

    print("=" * 70)
    print("SEARCH TEST — querying with key terms (no embedding, raw text search)")
    print("=" * 70)

    for name in CANDIDATE_COLLECTIONS:
        if name not in existing_collections:
            continue
        collection = client.get_collection(name)
        everything = collection.get(include=["documents"])
        docs = everything["documents"]

        print(f"\nIn collection '{name}':")
        for term in SEARCH_TERMS:
            count = sum(1 for d in docs if term in d)
            print(f"  '{term}' appears in {count}/{len(docs)} chunks")

    print(f"""

NEXT STEPS based on what you see above:

1. If halant density is HIGH (>8%) and/or the preview text looks
   visibly broken (like "এন্জজও" instead of "এনজিও") — the source
   document was embedded with corrupted text. This needs to be
   re-extracted cleanly BEFORE running the multilingual migration,
   otherwise you'll migrate the same corruption into the better
   embedding model and retrieval will still under-perform for this
   specific document.

2. If NO matches were found at all — the document may not have
   been successfully chunked/embedded in the first place, which is
   a separate ingestion problem to debug.

3. If the text looks clean — good, the corruption I flagged in the
   PDF extraction doesn't reflect what's actually in your database,
   and the multilingual migration alone should fix retrieval for
   this content.
""")


if __name__ == "__main__":
    main()