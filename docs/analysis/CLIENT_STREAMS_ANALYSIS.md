# Client-Side Stream Storage Analysis & Proposal

## Current Architecture

### How Client Stores Data

**Database:** SQLite (`/Satori/Engine/db/engine.db`)

**Current Storage Model:**
- Each **stream** = separate **table** (table name = streamUUID)
- Table schema: `ts, value, hash, provider`
- **NO streams metadata table** currently exists

**Example:**
```
Table: "abc123-def456-streamuuid"
  ts                | value      | hash    | provider
  2024-01-03 12:00  | 45000.50   | xyz123  | central
  2024-01-03 13:00  | 45100.25   | xyz124  | central
```

### How Client Fetches Observations

**Endpoint:** `GET /api/v1/observation/get`

**Old Response (before stream feature):**
```json
{
  "id": 123,
  "value": "45000.50",
  "observed_at": "1704326400",
  "hash": "abc123",
  "ts": "2024-01-03T12:00:00"
}
```

**New Response (with stream feature):**
```json
{
  "id": 123,
  "value": "45000.50",
  "observed_at": "1704326400",
  "hash": "abc123",
  "stream_id": 1,
  "stream": {
    "id": 1,
    "name": "bitcoin",
    "secondary": null,
    "meta": null,
    "description": null
  },
  "ts": "2024-01-03T12:00:00"
}
```

## The Problem

❌ **Client receives stream metadata but has nowhere to store it**

- Server now sends `stream_id` and full `stream` object
- Client currently just stores: timestamp, value, hash, provider
- Stream name, description, and other metadata are **lost**
- No way to query "what streams do I have?" or "what's the name of this stream?"

## Proposed Solution

### Add Streams Metadata Table

Create a `streams` table in the client SQLite database to store stream information.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS streams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_stream_id INTEGER,           -- Stream ID from central server
    uuid TEXT UNIQUE NOT NULL,          -- Stream UUID (used as table name)
    name TEXT,                          -- Stream name (e.g., "bitcoin")
    secondary TEXT,                     -- Secondary identifier
    meta TEXT,                          -- Metadata field
    description TEXT,                   -- Stream description
    last_synced TIMESTAMP,              -- Last time we synced this stream
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Updated Observation Storage Flow

**When client fetches observation from server:**

```python
# 1. Fetch observation
response = server.getObservation()
# {
#   "value": "45000.50",
#   "stream_id": 1,
#   "stream": {"id": 1, "name": "bitcoin", ...},
#   ...
# }

# 2. Extract stream metadata
if response.get('stream'):
    stream_metadata = response['stream']

    # 3. Store/update stream metadata in streams table
    db.upsertStream(
        server_stream_id=stream_metadata['id'],
        uuid=streamUUID,  # From existing logic
        name=stream_metadata['name'],
        secondary=stream_metadata.get('secondary'),
        meta=stream_metadata.get('meta'),
        description=stream_metadata.get('description')
    )

# 4. Store observation data in stream table (existing logic)
db.insertRow(streamUUID, timestamp, value, hash, provider)
```

### Benefits

✅ **Stream identification:** Know what each streamUUID table represents
✅ **Human-readable names:** Map "bitcoin" → streamUUID table
✅ **Metadata preservation:** Store descriptions and other stream info
✅ **Server synchronization:** Track which streams exist on server
✅ **Backward compatible:** Old observations without stream info still work

## Implementation Files

### Files to Create/Modify

1. **`engine-lite/storage/sqlite_manager.py`**
   - Add `createStreamsTable()` method
   - Add `upsertStream()` method
   - Add `getStreamByUuid()` method
   - Add `getStreamByName()` method
   - Add `getAllStreams()` method

2. **`engine-lite/storage/manager.py`**
   - Add stream metadata methods
   - Update `storeStreamObservation()` to accept stream metadata

3. **`lib-lite/satorilib/server/server.py`**
   - Update `getObservation()` to extract and store stream metadata

4. **Migration Script**
   - Create `migrations/add_client_streams_table.sql`

## Example Usage After Implementation

```python
# Get stream metadata
stream = db.getStreamByName("bitcoin")
# {
#   "id": 1,
#   "server_stream_id": 1,
#   "uuid": "abc123-def456",
#   "name": "bitcoin",
#   "description": None
# }

# List all streams client is tracking
streams = db.getAllStreams()
# [
#   {"uuid": "abc123", "name": "bitcoin"},
#   {"uuid": "def456", "name": "ethereum"},
# ]

# Get stream UUID by name (reverse lookup)
uuid = db.getStreamUuidByName("bitcoin")
# "abc123-def456"

# Then get data for that stream
data = db.getTableData(uuid)
```

## Migration Path

### For Existing Clients

1. **Create streams table** (new table, doesn't affect existing data)
2. **Backfill stream metadata:**
   - For streams with data but no metadata
   - Next time observation is fetched, stream metadata will be populated

3. **No data loss:**
   - Existing observation tables remain unchanged
   - Stream metadata is additive, not required

## Database Schema Comparison

### Current State
```
Database: engine.db
Tables:
  - "streamUUID1" (ts, value, hash, provider)
  - "streamUUID2" (ts, value, hash, provider)
  - ...
```

### Proposed State
```
Database: engine.db
Tables:
  - streams (id, server_stream_id, uuid, name, secondary, meta, description, last_synced, created_at)
  - "streamUUID1" (ts, value, hash, provider)
  - "streamUUID2" (ts, value, hash, provider)
  - ...
```

## Next Steps

1. ✅ Document current architecture (this file)
2. ⬜ Create streams table schema
3. ⬜ Implement database methods
4. ⬜ Update observation fetch/store logic
5. ⬜ Add tests
6. ⬜ Create migration script

## Questions to Consider

1. **Should we store server_stream_id or rely only on uuid?**
   - Recommendation: Store both for flexibility

2. **How to handle stream updates?**
   - Use `UPSERT` logic (INSERT OR REPLACE)
   - Update `last_synced` timestamp

3. **What if server stream changes?**
   - `last_synced` helps detect staleness
   - Could implement sync endpoint to refresh all streams

4. **Backward compatibility?**
   - Fully backward compatible
   - Streams table is optional metadata
   - Observation tables work independently
