"""Streaming Parsers"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Iterator, Dict, Any, List
import io


class JSONStreamParser:
    """Stream JSON objects one at a time.

    Perfect for interviews - demonstrates:
    - Memory-efficient parsing
    - Generator patterns
    - Handling large files
    """

    @staticmethod
    def parse_json_lines(file_path: str) -> Iterator[Dict[str, Any]]:
        """Parse JSONL (JSON Lines) format - one JSON object per line."""
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error on line {line_num}: {e}")

    @staticmethod
    def parse_json_array(file_path: str) -> Iterator[Dict[str, Any]]:
        """Parse large JSON array without loading everything into memory."""
        with open(file_path, 'r') as f:
            # Skip opening bracket
            char = f.read(1)
            while char and char.isspace():
                char = f.read(1)
            if char != '[':
                raise ValueError("Expected JSON array")

            decoder = json.JSONDecoder()
            buffer = ""
            bracket_count = 0
            in_string = False
            escape_next = False

            for char in f.read():
                buffer += char

                if not escape_next:
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Complete object found
                                try:
                                    obj = decoder.decode(buffer.strip().rstrip(','))
                                    yield obj
                                    buffer = ""
                                except json.JSONDecodeError:
                                    pass  # Continue accumulating
                    if char == '\\':
                        escape_next = True
                else:
                    escape_next = False


class CSVStreamParser:
    """Stream CSV rows efficiently."""

    @staticmethod
    def parse_csv(file_path: str, has_header: bool = True) -> Iterator[Dict[str, str]]:
        """Parse CSV file row by row."""
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f) if has_header else csv.reader(f)
            for row in reader:
                yield row

    @staticmethod
    def parse_large_csv(file_path: str, chunk_size: int = 1000) -> Iterator[List[Dict[str, str]]]:
        """Parse CSV in chunks for batch processing."""
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:  # Yield remaining rows
                yield chunk


class XMLStreamParser:
    """Stream XML elements efficiently."""

    @staticmethod
    def parse_xml_elements(file_path: str, tag: str) -> Iterator[ET.Element]:
        """Parse specific XML elements without loading entire document."""
        # Use iterparse for memory-efficient parsing
        for event, elem in ET.iterparse(file_path, events=('start', 'end')):
            if event == 'end' and elem.tag == tag:
                yield elem
                elem.clear()  # Free memory

    @staticmethod
    def parse_xml_to_dict(file_path: str, tag: str) -> Iterator[Dict[str, Any]]:
        """Parse XML elements and convert to dictionaries."""
        for elem in XMLStreamParser.parse_xml_elements(file_path, tag):
            yield XMLStreamParser._element_to_dict(elem)

    @staticmethod
    def _element_to_dict(elem: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if elem.attrib:
            result.update(elem.attrib)

        # Add text content
        if elem.text and elem.text.strip():
            if len(elem) == 0:  # No children, just text
                return elem.text.strip()
            result['text'] = elem.text.strip()

        # Add children
        for child in elem:
            child_data = XMLStreamParser._element_to_dict(child)
            if child.tag in result:
                # Multiple children with same tag - make it a list
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result


class LogStreamParser:
    """Parse log files efficiently - common interview topic."""

    @staticmethod
    def parse_access_logs(file_path: str) -> Iterator[Dict[str, str]]:
        """Parse Apache/Nginx access logs."""
        # Common log format: IP - - [timestamp] "method path protocol" status size
        import re
        log_pattern = re.compile(
            r'(\S+) - - \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)'
        )

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                match = log_pattern.match(line)
                if match:
                    yield {
                        'ip': match.group(1),
                        'timestamp': match.group(2),
                        'method': match.group(3),
                        'path': match.group(4),
                        'protocol': match.group(5),
                        'status': int(match.group(6)),
                        'size': int(match.group(7)),
                        'line_number': line_num
                    }

    @staticmethod
    def filter_logs(file_path: str, status_code: int = None,
                   ip_address: str = None) -> Iterator[Dict[str, str]]:
        """Filter logs by criteria."""
        for log_entry in LogStreamParser.parse_access_logs(file_path):
            if status_code and log_entry['status'] != status_code:
                continue
            if ip_address and log_entry['ip'] != ip_address:
                continue
            yield log_entry


def create_sample_files():
    """Create sample files for demonstration."""
    # Create sample JSONL file
    with open('sample.jsonl', 'w') as f:
        f.write('{"id": 1, "name": "Alice", "age": 25}\n')
        f.write('{"id": 2, "name": "Bob", "age": 30}\n')
        f.write('{"id": 3, "name": "Charlie", "age": 35}\n')

    # Create sample CSV file
    with open('sample.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'department', 'salary'])
        writer.writerow([1, 'Alice', 'Engineering', 75000])
        writer.writerow([2, 'Bob', 'Sales', 65000])
        writer.writerow([3, 'Charlie', 'Marketing', 70000])

    # Create sample XML file
    with open('sample.xml', 'w') as f:
        f.write('''<?xml version="1.0"?>
<employees>
    <employee id="1">
        <name>Alice</name>
        <department>Engineering</department>
        <salary>75000</salary>
    </employee>
    <employee id="2">
        <name>Bob</name>
        <department>Sales</department>
        <salary>65000</salary>
    </employee>
</employees>''')

    # Create sample log file
    with open('sample.log', 'w') as f:
        f.write('192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234\n')
        f.write('192.168.1.2 - - [01/Jan/2024:12:00:01 +0000] "POST /api/login HTTP/1.1" 401 567\n')
        f.write('192.168.1.1 - - [01/Jan/2024:12:00:02 +0000] "GET /api/data HTTP/1.1" 500 890\n')


def demonstrate_streaming_parsers():
    """Demonstrate streaming parser implementations."""
    print("=== Streaming Parsers Demo ===")

    # Create sample files
    create_sample_files()

    # JSON Lines parsing
    print("\n1. JSON Lines Parsing:")
    for obj in JSONStreamParser.parse_json_lines('sample.jsonl'):
        print(f"  {obj}")

    # CSV parsing
    print("\n2. CSV Parsing:")
    for row in CSVStreamParser.parse_csv('sample.csv'):
        print(f"  {row}")

    # CSV chunked parsing
    print("\n3. CSV Chunked Parsing:")
    for chunk in CSVStreamParser.parse_large_csv('sample.csv', chunk_size=2):
        print(f"  Chunk: {chunk}")

    # XML parsing
    print("\n4. XML Parsing:")
    for employee in XMLStreamParser.parse_xml_to_dict('sample.xml', 'employee'):
        print(f"  {employee}")

    # Log parsing
    print("\n5. Log Parsing:")
    for log_entry in LogStreamParser.parse_access_logs('sample.log'):
        print(f"  {log_entry['ip']} -> {log_entry['method']} {log_entry['path']} ({log_entry['status']})")

    # Filtered log parsing
    print("\n6. Filtered Logs (status 200):")
    for log_entry in LogStreamParser.filter_logs('sample.log', status_code=200):
        print(f"  {log_entry['ip']} -> {log_entry['path']}")

    # Memory efficiency demonstration
    print("\n7. Memory Efficiency:")
    print("✓ Parsers use generators - process one item at a time")
    print("✓ Can handle files larger than available RAM")
    print("✓ Constant memory usage regardless of file size")

    # Clean up
    import os
    for filename in ['sample.jsonl', 'sample.csv', 'sample.xml', 'sample.log']:
        try:
            os.remove(filename)
        except:
            pass


def interview_questions():
    """Common streaming parser interview questions."""
    print("\n=== Interview Q&A ===")

    print("\nQ: Why use streaming parsers?")
    print("A: Memory efficiency - can process files larger than RAM")

    print("\nQ: What's the time complexity?")
    print("A: O(n) where n is file size, but O(1) memory per item")

    print("\nQ: How to handle malformed data?")
    print("A: Try-catch blocks, skip bad records, log errors with line numbers")

    print("\nQ: When not to use streaming?")
    print("A: When you need random access or multiple passes over data")

    print("\nQ: Alternative approaches?")
    print("A: Pandas chunks, memory mapping, database streaming")


if __name__ == "__main__":
    demonstrate_streaming_parsers()
    interview_questions()