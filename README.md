# Install

```
brew install poppler
pip install -r requirements.txt
```

## Start Neo4J

```
brew services start neo4j
open http://localhost:7474
```

Or start Neo4j Desktop and go from there.
You need APOC installed.

## Extract Graph from PDF

```
python pdf_reader.py <pdf-path>
```

## Query Graph

```
python query.py
```
