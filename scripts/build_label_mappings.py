#!/usr/bin/env python3
"""Helper to generate label_mappings.json by listing VNClassify's taxonomy.

Run this on macOS to extract all known VNClassify labels, then manually map
each one to a child vocabulary word (or null to reject).

Usage: python3 build_label_mappings.py > vn_labels.txt
Then manually create label_mappings.json from the output.
"""

import subprocess
import json

# This script runs a Swift snippet to extract VNClassify labels
swift_code = '''
import Vision
let request = VNClassifyImageRequest()
let ids = try! request.supportedIdentifiers()
for id in ids.sorted() {
    print(id)
}
'''

print("Run the following in a Swift playground or command line to get all VNClassify labels:")
print("---")
print(swift_code)
print("---")
print("Then map each label to your vocabulary word or null in label_mappings.json")
