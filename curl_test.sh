#!/bin/sh

curl -X POST "http://211.229.2.155:8000/api/1.0.0/get_diagnosis" \
		 -H "Content-Type: application/json" \
		 -d '{
			 "features": [
				 "Spondylosis.",
				 "OLF, T9-11 Broad based central extrusion, T11-12 Mild central stenosis, L1-2 Central downward extrusion, moderate central stenosis, L2-3 Mild central stenosis, L3-4 Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5 Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1",
				 "OLF, T9-11. Broad based central extrusion, T11-12. Mild central stenosis, L1-2. Central downward extrusion, moderate central stenosis, L2-3. Mild central stenosis, L3-4. Broad based central~Lt foraminal extrusion w/ LRS & NFS, severe central stenosis, L4-5. Central~Lt foraminal extrusion w/ LRS & NFS, L5-S1",
				 7, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 90, 90, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
			 ]
		 }'
