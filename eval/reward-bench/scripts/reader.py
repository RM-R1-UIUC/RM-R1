file_path = "/shared/nas2/xiusic/Rubric-RM/document_guideline/model_spec_chat.txt"

with open(file_path, "r", encoding="utf-8") as file:
    # Read the entire file into a single string
    content = file.read()

print(content)