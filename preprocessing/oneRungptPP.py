# This file allows the chatGpt preprocessing (Question/Answer Generation) from the console
import json
import os
from openai import OpenAI
import time

def is_file_over_limit(file_path, size_limit_mb=80):
    return os.path.getsize(file_path) >= size_limit_mb * 1024 * 1024

definition_string_question = """
Use german language.
Please generate a question asking for the key information in the given input.
Please ask the specific question instead of the general question, like
'What is the key information in the given paragraph?'.
"""

# Generation for question gpt request corpus
def gpt_obj_adjusting_question(input_file_paths, output_file_path):
    all_data = []
    entry_id_count = 0
    file_count = 1
    current_file_path = f"{output_file_path}_{file_count}.jsonl"

    # Read the input JSON file
    for input_file_path in input_file_paths:
        print("Loaded... ", input_file_path)
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
            if isinstance(data, list):  # If data is a list, extend the list
                all_data.extend(data)
            else:  # If data is not a list, append it as an element
                all_data.append(data)

    # Write the updated data to the output JSON file
    with open(current_file_path, 'w') as output_file:
        for entry in all_data:
            # Generate first entry for question
            entry["custom_id"] = "question-" + str(entry_id_count)
            entry["method"] = "POST"
            entry["url"] = "/v1/chat/completions"
            entry["body"] = {"model": "gpt-3.5-turbo-0125",
                            "messages": [{"role": "system", "content": definition_string_question}, # You are a helpful assistant.
                                            {"role": "user", "content": f"{entry['input']}; {entry['output']}"}],
                            "max_tokens": 2000}
            entry_id_count += 1
            del entry['input']
            del entry['output']
            if is_file_over_limit(current_file_path):
                # Close the current file and open a new one
                output_file.close()
                file_count += 1
                current_file_path = f"{output_file_path}_{file_count}.jsonl"
                output_file = open(current_file_path, 'w')
            json.dump(entry, output_file)
            output_file.write('\n')

    return file_count

definition_string_answer = """
Use german language.
Answer the provided question only using the information in the given context.
Please generate the answer using as much information as possible, but dont imagine any additional information.
Don't add any information and just focus on the context you receive.
The answer should be informative and should be more than 3 sentences.
"""

# Generation for answer
def gpt_obj_adjusting_answer(gpt_results, reference_array, output_file_path):
    all_data = []
    count = 0

    # Write the updated data to the output JSON file
    with open(output_file_path, 'w') as output_file:
        for result_entry in gpt_results:
            # Get corresponding obj from reference_array
            correspond_obj = next((item for item in reference_array if item["custom_id"] == result_entry["custom_id"]), None)

            if correspond_obj:
                entry = {}
                # Generate first entry for question
                entry["custom_id"] = correspond_obj["custom_id"]
                entry["method"] = "POST"
                entry["url"] = "/v1/chat/completions"
                entry["body"] = {"model": "gpt-3.5-turbo-0125",
                                "messages": [{"role": "system", "content": definition_string_answer},
                                            {"role": "user", "content": f"Question: {result_entry['question']}; Context: {correspond_obj['data']}"}],
                                "max_tokens": 2000}
                json.dump(entry, output_file)
                output_file.write('\n')
                count += 1
    return count

# Function to wait for the file creation to complete
def wait_for_file_creation(file_id, polling_interval=1):
    isSuccess = False
    while True:
        # Retrieve the file status
        file_status = client.files.retrieve(file_id)
        
        # Check if the file is ready (depending on the API's status response)
        if file_status.status == 'processed':
            isSuccess = True
            break
        elif file_status.status == 'failed':
            isSuccess = False
            break
        else:
            print(f"File status: {file_status.status} -> Waiting...")

        # Wait for the specified polling interval before checking again
        time.sleep(polling_interval)

    return isSuccess

# Function to poll a job status (120 -> 2 minutes)
def wait_for_batch_job(batch_job_id, polling_interval=120):
    isSuccess = False

    while True:
        # Retrieve the batch job status
        batch_job = client.batches.retrieve(batch_job_id)

        # Check if the job is completed or failed
        if batch_job.status == 'completed':
            isSuccess = True
            break
        elif batch_job.status == 'failed':
            isSuccess = False
            break
        else:
            print(f"Batch job status: {batch_job.status}. Waiting... -> {batch_job.request_counts}")
        
        # Wait for the specified polling interval before checking again
        time.sleep(polling_interval)

    print("Final batcj job counts: ", batch_job.request_counts)

    return isSuccess

client = OpenAI(
    api_key="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"
)

# 1) QUESTION GENERATION

# Specify the path to input JSON file
# input_file_paths = ["allSR/bb.json", "allSR/be.json", "allSR/bund.json", "allSR/he.json", "allSR/hh.json", "allSR/mv.json", "allSR/rp.json", "allSR/sh.json", "allSR/sl.json", "allSR/st.json", "allSR/th.json"]
jus_prefix = "bund"
input_file_paths = [f"allSR/{jus_prefix}.json"]
output_file_path_questions = f"generatedData/{jus_prefix}_adjustedForGpt_questions"

# Iterates over given json files adjustes them for chatgpt and saves them to a jsonl file
file_count = gpt_obj_adjusting_question(input_file_paths, output_file_path_questions)
print("ChatGpt objects have been creaated and were saved to jsonl files")
print("Amount of files ",file_count)

for i in range(file_count):
    file_number = i + 1
    adjustedQuestionsFilename = f"generatedData/{jus_prefix}_adjustedForGpt_questions_{file_number}.jsonl"
    result_file_name_questions = f"generatedData/{jus_prefix}_gptResults_questions_{file_number}.jsonl"

    adjustedAnswersFilename = f"generatedData/{jus_prefix}_adjustedForGpt_answers_{file_number}.jsonl"
    result_file_name_answer = f"generatedData/{jus_prefix}_gptResults_answers_{file_number}.jsonl"

    final_file = f"results/{jus_prefix}_preprocessingData_{file_number}.json"

    # Generate Batch for chatgpt
    batch_input_file = client.files.create(
    file=open(adjustedQuestionsFilename, "rb"),
    purpose="batch"
    )

    print("Started batch file creation for ", adjustedQuestionsFilename)
    isSuccess = wait_for_file_creation(batch_input_file.id)

    if isSuccess:
        print("Finished batch file creation for ", adjustedQuestionsFilename)

        # Start Batch job
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "question generation job"
            }
        )

        print("StartedBatchJob: ", batch_job.id, " for ", adjustedQuestionsFilename)
        isSuccess = wait_for_batch_job(batch_job.id)

        if isSuccess:
            print("Finished Batch job for ", adjustedQuestionsFilename)
            batch_job = client.batches.retrieve(batch_job.id)

            print("Retrieve output file: ", batch_job.output_file_id)
            result_questions = client.files.content(batch_job.output_file_id).content

            # Save results as jsonl file
            with open(result_file_name_questions, 'wb') as file:
                file.write(result_questions)

            # Loading data from saved jsonl file
            results_question_parsed= []
            with open(result_file_name_questions, 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results_question_parsed.append(json_object)

            final_data_questions = []
            for res in results_question_parsed:
                result = res['response']['body']['choices'][0]['message']['content']
                id = res["custom_id"]
                final_data_questions.append({'custom_id': id, 'question': result})

            print(result_file_name_questions, " -> Final data length: ", len(final_data_questions))

            # 2) ANSWER GENERATION

            # Generated references from Jsonl to reicive the corresponding input and output
            reference_object_questions = []
            with open(adjustedQuestionsFilename, 'r') as file:
                for line in file:
                    json_object = json.loads(line.strip())
                    reference_object_questions.append({"custom_id": json_object["custom_id"], "data": json_object['body']['messages'][1]['content']})

            print(adjustedQuestionsFilename, " -> references length: ", len(reference_object_questions))

            count = gpt_obj_adjusting_answer(final_data_questions, reference_object_questions, adjustedAnswersFilename)
            print("Wrote ", count, " objects into ", adjustedAnswersFilename)

            batch_input_file = client.files.create(
                file=open(adjustedAnswersFilename, "rb"),
                purpose="batch"
            )

            print("Started batch file creation for ", adjustedAnswersFilename)
            isSuccess = wait_for_file_creation(batch_input_file.id)

            if isSuccess:
                print("Finished batch file creation for ", adjustedAnswersFilename)

                batch_job = client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                    "description": "nightly eval job"
                    }
                )

                print("StartedBatchJob: ", batch_job.id, " for ", adjustedAnswersFilename)
                isSuccess = wait_for_batch_job(batch_job.id)

                if isSuccess:
                    print("Finished Batch job for ", adjustedAnswersFilename)

                    batch_job = client.batches.retrieve(batch_job.id)
                    result = client.files.content(batch_job.output_file_id).content

                    with open(result_file_name_answer, 'wb') as file:
                        file.write(result)

                    # Loading data from saved file
                    results = []
                    with open(result_file_name_answer, 'r') as file:
                        for line in file:
                            # Parsing the JSON string into a dict and appending to the list of results
                            json_object = json.loads(line.strip())
                            results.append(json_object)

                    final_data_answers = []
                    for res in results:
                        result = res['response']['body']['choices'][0]['message']['content']
                        id = res["custom_id"]
                        final_data_answers.append({'custom_id': id, 'answer': result})

                    print(result_file_name_answer, " -> Final data length: ", len(final_data_answers))


                    # 3) ADD BOTH TOGETHER

                    # Generate training data
                    training_data = []
                    for result_q in final_data_questions:
                        # Get corresponding obj from reference_array
                        correspond_obj = next((item for item in final_data_answers if item["custom_id"] == result_q["custom_id"]), None)

                        if correspond_obj:
                            training_data.append({"input": result_q["question"], "output": correspond_obj["answer"]})

                    print("Finished length: ", len(training_data))

                    # Saving the data to a JSON file
                    with open(final_file, 'w') as f:
                        json.dump(training_data, f)

                else:
                    print("Failed Batch job for ", adjustedAnswersFilename, " continuing with the next jsonl")
            else:
                print("Batch input file creation failed for ", adjustedAnswersFilename, " continuing with the next jsonl")
        else:
            print("Failed Batch job for ", adjustedQuestionsFilename, " continuing with the next jsonl")
    else:
        print("Batch input file creaation failed for ", adjustedQuestionsFilename, " continuing with the next jsonl")
