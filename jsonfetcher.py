import openai
import json

def test_fetch():
    prompt = '''
    Create a valid json object of 5 questions to test dyscalculia. Make sure that the question vary from number based to word based so that the question looks authentic to dyslexic person. The output should be in a json format like:
        [{
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option"
        }]
        There should not be additional information in the output.
    '''

    # Define the messages with role-based structure
    messages = [
        {"role": "system", "content": "You are a helpful and creative assistant."},
        {"role": "user", "content": prompt}
    ]

    # Call the OpenAI API using Chat Completion with role-based messages
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=5000,  # Limit the response length
        temperature=0.7  # Controls randomness
    )

    # Extract and print the generated text
    generated_text = response['choices'][0]['message']['content'].strip()
    # print(generated_text)
    json_output = json.loads(generated_text)


    with open('testquestions.json', 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

    # print(json_output)
    return json_output


def Numeric_fetch():
    prompt = '''
    Create a valid json object of 10 questions to test dyscalculia. Make sure that the questions are Numerical BODMAS problems so that the question looks authentic to dyscalculic person. The output should be in a json format like:
        [{
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        }]
        There should not be additional information in the output.
    '''

    # Define the messages with role-based structure
    messages = [
        {"role": "system", "content": "You are a helpful and creative assistant."},
        {"role": "user", "content": prompt}
    ]

    # Call the OpenAI API using Chat Completion with role-based messages
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=10000,  # Limit the response length
        temperature=0.7  # Controls randomness
    )

    # Extract and print the generated text
    generated_text = response['choices'][0]['message']['content'].strip()
    # print(generated_text)
    json_output = json.loads(generated_text)

    with open('questions.json', 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

    # print(json_output)
    return json_output

def WordFetch():
    prompt = '''
    Create a valid json object of 10 questions to test dyscalculia. Make sure that the question word based mathematical problems so that the question that is difficult to a dyscalculic person. The output should be in a json format like:
        [{
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        },
        {
        "question ": "example question",
        "options": [ list of strings with one correct answer],
        "correct": "the correct answer form the option",
        "explanation": "The explanation of the answer"
        }]
        There should not be additional information in the output.
    '''

    # Define the messages with role-based structure
    messages = [
        {"role": "system", "content": "You are a helpful and creative assistant who only give output in a valid json format."},
        {"role": "user", "content": prompt}
    ]

    # Call the OpenAI API using Chat Completion with role-based messages
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=5000,  # Limit the response length
        temperature=0.7  # Controls randomness
    )

    # Extract and print the generated text
    generated_text = response['choices'][0]['message']['content'].strip()

    print(generated_text)
    # print(generated_text)
    json_output = json.loads(generated_text)

    with open('wordquestions.json', 'w') as json_file:
        json.dump(json_output, json_file, indent=4)
    print(json_output)

    # print(json_output)
    return json_output