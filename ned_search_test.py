import query
from query import hyperleda

result = hyperleda.query_object(
    "NGC 7172",
    properties="all"
)

print(result.colnames)
print(result)