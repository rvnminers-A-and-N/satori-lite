# Stream UUID Mismatch Analysis

## The Problem

**Client** and **Server** have **completely different** stream identification systems!

### Server Side (Central-Lite)

**Stream table:**
```sql
CREATE TABLE streams (
    id INTEGER,
    name VARCHAR(255),  -- e.g., "bitcoin", "ethereum"
    ...
)
```

**Observation response:**
```json
{
  "value": "45000.50",
  "stream_id": 1,
  "stream": {
    "id": 1,
    "name": "bitcoin"
  }
}
```

### Client Side (Satori-Lite)

**StreamId components:**
```python
StreamId(
    source="central",      # Where the data comes from
    author="satori",            # Who publishes it
    stream="observations",      # Stream type
    target=""                   # Specific target field
)
```

**Generated UUID:**
```python
# UUID5 from: "central:satori:observations:"
uuid = "abc123-def456-789abc-..."
```

**Database:**
```
Table name: "abc123-def456-789abc-..."
  ts                | value      | hash    | provider
  2024-01-03 12:00  | 45000.50   | xyz123  | central
```

## How Client Generates streamUUID

**Code:** `lib-lite/satorilib/concepts/structs.py`

```python
class StreamId:
    def __init__(self, source: str, author: str, stream: str, target: str = ''):
        self.__source = source
        self.__author = author
        self.__stream = stream
        self.__target = target

    @property
    def uuid(self) -> str:
        return str(StreamId.generateUUID(self))

    @staticmethod
    def generateUUID(data) -> uuid:
        namespace = uuid.NAMESPACE_DNS
        # Concatenate: "source:author:stream:target"
        values = [data.source, data.author, data.stream, data.target]
        data_str = ':'.join(str(v) for v in values)
        return uuid.uuid5(namespace, data_str)
```

**Example:**
```python
streamId = StreamId(
    source="central",
    author="satori",
    stream="observations",
    target=""
)
# uuid = UUID5("central:satori:observations:")
# Result: "7a8b9c0d-1e2f-5a6b-c7d8-e9f0a1b2c3d4" (example)
```

## The Mismatch

### What Server Knows
- Stream name: **"bitcoin"**
- Stream ID: **1**
- Stream metadata: description, secondary, meta

### What Client Knows
- StreamId components: **source="central", author="satori", stream="observations", target=""**
- StreamUUID: **"7a8b9c0d-1e2f-5a6b-c7d8-e9f0a1b2c3d4"**
- No knowledge of "bitcoin" vs "ethereum" distinction

### The Problem

**All observations go to the same client table!**

Whether the observation is for Bitcoin, Ethereum, or any other crypto, the client stores it in the same table because:

1. Client generates streamUUID from hardcoded components
2. Components are always the same: `("central", "satori", "observations", "")`
3. Same components → same UUID → same table

**Client cannot distinguish between different observation streams!**

## Current Behavior

### Server sends Bitcoin observation:
```json
{
  "value": "45000.50",
  "stream": {"name": "bitcoin"}
}
```

### Client receives and stores:
```python
# Client generates streamUUID from hardcoded components
streamId = StreamId("central", "satori", "observations", "")
uuid = streamId.uuid  # Same UUID every time!

# Stores in table named by UUID
db.insertRow(
    table_uuid=uuid,  # "7a8b9c0d-..."
    timestamp="2024-01-03 12:00",
    value="45000.50",
    ...
)
```

### Server sends Ethereum observation:
```json
{
  "value": "2500.00",
  "stream": {"name": "ethereum"}
}
```

### Client receives and stores:
```python
# SAME streamUUID generated!
streamId = StreamId("central", "satori", "observations", "")
uuid = streamId.uuid  # SAME UUID as Bitcoin!

# Stores in SAME table
db.insertRow(
    table_uuid=uuid,  # "7a8b9c0d-..." (SAME!)
    timestamp="2024-01-03 12:00",
    value="2500.00",
    ...
)
```

**Result:** Bitcoin and Ethereum prices mixed together in one table! ❌

## Solutions

### Option 1: Include Stream Name in StreamId Components

**Modify client to use server's stream name in StreamId:**

```python
# Before (all observations → same table)
streamId = StreamId(
    source="central",
    author="satori",
    stream="observations",  # Generic
    target=""
)

# After (each stream → different table)
streamId = StreamId(
    source="central",
    author="satori",
    stream=server_stream_name,  # "bitcoin", "ethereum", etc.
    target=""
)
```

**Impact:**
- Different stream names → different UUIDs → different tables
- Bitcoin: `StreamId(..., stream="bitcoin")` → table "abc123-..."
- Ethereum: `StreamId(..., stream="ethereum")` → table "def456-..."

**Code change needed:**
```python
# In server.py or wherever observation is processed
response = getObservation()
if response.get('stream'):
    stream_name = response['stream']['name']

    # Use stream name in StreamId
    streamId = StreamId(
        source="central",
        author="satori",
        stream=stream_name,  # ← Use server's stream name
        target=""
    )
```

### Option 2: Use Server Stream UUID Directly

**Have server generate and send the streamUUID:**

**Server adds to response:**
```json
{
  "value": "45000.50",
  "stream": {
    "name": "bitcoin",
    "uuid": "7a8b9c0d-1e2f-5a6b-c7d8-e9f0a1b2c3d4"  ← Server provides UUID
  }
}
```

**Client uses server's UUID directly:**
```python
response = getObservation()
stream_uuid = response['stream']['uuid']  # Use server's UUID

# Store using server-provided UUID
db.insertRow(
    table_uuid=stream_uuid,
    ...
)
```

**Impact:**
- Client doesn't generate UUID, just uses server's
- Server controls stream identification
- Centralized stream management

### Option 3: Hybrid Approach (Recommended)

**Use both systems:**

1. **Server sends stream metadata** (name, UUID, description)
2. **Client stores metadata** in `streams` table
3. **Client generates streamUUID** from components that include stream name
4. **Client maps** server stream → local streamUUID

**Implementation:**

```python
# Server response
response = {
    "value": "45000.50",
    "stream": {
        "id": 1,
        "name": "bitcoin",
        "description": "Bitcoin price"
    }
}

# Client processing
stream_name = response['stream']['name']

# Generate streamUUID using server's stream name
streamId = StreamId(
    source="central",
    author="satori",
    stream=stream_name,  # ← Key: use server's stream name
    target=""
)
stream_uuid = streamId.uuid

# Store stream metadata mapping
db.upsertStream(
    uuid=stream_uuid,
    server_stream_id=response['stream']['id'],
    name=stream_name,
    description=response['stream'].get('description')
)

# Store observation in stream-specific table
db.insertRow(
    table_uuid=stream_uuid,
    ...
)
```

## Recommended Action

**Option 3 (Hybrid Approach)** is best because:

✅ Preserves client's existing StreamId system
✅ Leverages server's stream metadata
✅ Each crypto gets its own table
✅ Backward compatible (can handle old observations without stream)
✅ Maintains deterministic UUID generation
✅ Stores stream metadata for reference

## Code Changes Required

### 1. Update Client Observation Fetching

**File:** `lib-lite/satorilib/server/server.py`

```python
def getObservation(self, stream: str = 'bitcoin') -> Union[dict, None]:
    response = self._makeAuthenticatedCall(...)
    data = response.json()

    # NEW: Extract stream name and use it to generate streamId
    if data and data.get('stream'):
        stream_name = data['stream'].get('name')

        # Generate streamId using server's stream name
        self.streamId = StreamId(
            source="central",
            author="satori",
            stream=stream_name,  # ← Use server's stream name
            target=""
        )

        # Store stream metadata
        self.storage.db.upsertStream(
            uuid=self.streamId.uuid,
            server_stream_id=data['stream'].get('id'),
            name=stream_name,
            secondary=data['stream'].get('secondary'),
            meta=data['stream'].get('meta'),
            description=data['stream'].get('description')
        )

    return data
```

### 2. Update Stream Storage

**File:** `engine-lite/engine.py`

Ensure that when storing observations, the streamUUID is generated from the stream-specific StreamId (which now includes the stream name).

## Testing

```python
# Test different streams generate different UUIDs
bitcoin_id = StreamId("central", "satori", "bitcoin", "")
ethereum_id = StreamId("central", "satori", "ethereum", "")

print(f"Bitcoin UUID: {bitcoin_id.uuid}")
print(f"Ethereum UUID: {ethereum_id.uuid}")

# Should be different!
assert bitcoin_id.uuid != ethereum_id.uuid
```

## Summary

**Current Problem:** Client generates same streamUUID for all observations, mixing different data streams in one table.

**Root Cause:** StreamId components are hardcoded and don't include server's stream name.

**Solution:** Use server's stream name in StreamId.stream component to generate unique UUIDs per stream.

**Benefit:** Each crypto gets its own table, proper data separation, while maintaining existing StreamId system.
