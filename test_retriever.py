
from retriever import retrieve_documents

# Test 1: Empty documents
result = retrieve_documents(
    [0.1, 0.2],
    [],
    [],
    top_k=3
)

assert result == []
print("Test 1 passed: Empty documents")

# Test 2: Invalid top_k
try:
    retrieve_documents(
        [0.1, 0.2],
        ["Document A"],
        [[0.1, 0.2]],
        top_k=0
    )
except ValueError:
    print("Test 2 passed: Invalid top_k")

# Test 3: Invalid threshold
try:
    retrieve_documents(
        [0.1, 0.2],
        ["Document A"],
        [[0.1, 0.2]],
        top_k=1,
        threshold=1.5
    )
except ValueError:
    print("Test 3 passed: Invalid threshold")
