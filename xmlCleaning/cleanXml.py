import xml.etree.ElementTree as ET
import json
import re

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def extract_text_elements(root):
    elements = []
    for elem in root.findall('.//p'):
        text = elem.text
        if text:
            elements.append(text)
    return elements

def clean_elment(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_all(texts):
    return [clean_elment(text) for text in texts]

def save_to_json(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
        
        
        
# Single file
#xml_file = 'testXML.xml'
#xml_root = parse_xml_file(xml_file)
#
#text_elements = extract_text_elements(xml_root)
#print(text_elements)
#
#cleaned_text_elements = clean_all(text_elements)
#print(cleaned_text_elements)
#
#output_file = 'data.json'
#save_to_json(cleaned_text_elements, output_file)



# Multiple files
xml_files = ['testXML.xml', 'test2.xml', 'test3.xml']
result_list = []

for file in xml_files:
    xml_root = parse_xml_file(file)
    text_elements = extract_text_elements(xml_root)
    cleaned_text_elements = clean_all(text_elements)
    result_list += cleaned_text_elements
    
output = 'data2.json'
save_to_json(result_list, output)
