-- SQL script to create the 'earthquakes' table in PostgreSQL

CREATE TABLE earthquakes (
    time TIMESTAMPTZ,               -- Event time
    latitude DOUBLE PRECISION,      -- Epicenter latitude
    longitude DOUBLE PRECISION,     -- Epicenter longitude
    depth DOUBLE PRECISION,         -- Depth in km
    mag DOUBLE PRECISION,           -- Magnitude
    magType VARCHAR(10),            -- Magnitude type
    nst DOUBLE PRECISION,           -- Station count
    gap DOUBLE PRECISION,           -- Azimuthal gap
    dmin DOUBLE PRECISION,          -- Nearest station distance
    rms DOUBLE PRECISION,           -- Travel time RMS
    net VARCHAR(10),                -- Reporting network
    id VARCHAR(30) PRIMARY KEY,     -- Event ID
    updated TIMESTAMPTZ,            -- Last update
    place TEXT,                     -- Location description
    type VARCHAR(20),               -- Event type
    horizontalError DOUBLE PRECISION,  -- Horizontal error
    depthError DOUBLE PRECISION,       -- Depth error
    magError DOUBLE PRECISION,         -- Magnitude error
    magNst DOUBLE PRECISION,           -- Stations for mag
    status VARCHAR(20),               -- Review status
    locationSource VARCHAR(10),       -- Location source
    magSource VARCHAR(10),            -- Magnitude source
    year INTEGER,                     -- Event year
    month INTEGER,                    -- Event month
    day INTEGER,                      -- Event day
    hour INTEGER                      -- Event hour
);

