import requests
import time
import json
from fuzzywuzzy import fuzz # Import the fuzzywuzzy library

API_URL = "http://localhost:8000/api/v1/hackrx/run"  # Change to Render URL if deployed
AUTH_TOKEN = "ae3c781e80a6b6d0ec74b60585efe1b7c06c17b09b6b332f076692de6dcfd64b"  # Replace with your actual token

# Payload
payload = {
    "documents": "https://hackrx.in/policies/ICIHLIP22012V012223.pdf",
    
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
questions_data = [
    {
        "question": "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "expected_answer": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
    },
    {
        "question": "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "expected_answer": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    },
    {
        "question": "Does this policy cover maternity expenses, and what are the conditions?",
        "expected_answer": "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
    },
    {
        "question": "What is the waiting period for cataract surgery?",
        "expected_answer": "The policy has a specific waiting period of two (2) years for cataract surgery."
    },
    {
        "question": "Are the medical expenses for an organ donor covered under this policy?",
        "expected_answer": "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
    },
    {
        "question": "What is the No Claim Discount (NCD) offered in this policy?",
        "expected_answer": "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium."
    },
    {
        "question": "Is there a benefit for preventive health check-ups?",
        "expected_answer": "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits."
    },
    {
        "question": "How does the policy define a 'Hospital'?",
        "expected_answer": "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients."
    },
    {
        "question": "What is the extent of coverage for AYUSH treatments?",
        "expected_answer": "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
    },
    {
        "question": "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "expected_answer": "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    }
]

# Extract only the questions for the API payload
questions_for_api = [item["question"] for item in questions_data]


# Headers for the API request, including Authorization
headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def test_api():
    """
    Tests the HackRx API by sending questions and comparing actual answers
    with expected answers using fuzzy matching. It then prints a detailed report
    and the overall average similarity percentage.
    """
    print("Testing HackRx API...")
    start_time = time.time()

    try:
        # Send POST request to the API
        response = requests.post(API_URL, headers=headers, json=payload)
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            actual_answers = data.get("answers", [])

            print(f"\nResponse Time: {elapsed_time:.2f} seconds")
            print("\n--- Answer Comparison Report ---")

            total_similarity_score = 0
            total_questions = len(questions_data)

            # Iterate through each question, compare answers, and print results
            for i, item in enumerate(questions_data):
                question = item["question"]
                expected_answer = item["expected_answer"]
                actual_answer = actual_answers[i] if i < len(actual_answers) else "N/A (No answer received)"

                # Use fuzzy matching for comparison
                # token_sort_ratio handles reordered words and partial matches well.
                similarity_score = fuzz.token_sort_ratio(expected_answer, actual_answer)
                total_similarity_score += similarity_score # Accumulate scores

                print(f"\nQ{i+1}: {question}")
                print(f"  Expected: {expected_answer}")
                print(f"  Actual:   {actual_answer}")
                print(f"  Similarity Score: {similarity_score:.2f}%")

            # Calculate and print overall average similarity
            average_similarity_percentage = (total_similarity_score / total_questions) if total_questions > 0 else 0
            print("\n--- Overall Average Similarity ---")
            print(f"Total Questions: {total_questions}")
            print(f"Average Similarity: {average_similarity_percentage:.2f}%")

        else:
            # Handle API errors
            print(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"API test failed: Could not connect to the API at {API_URL}. Please ensure the API is running and accessible.")
    except json.JSONDecodeError:
        print("API test failed: Could not decode JSON response. The API might have returned invalid JSON.")
    except Exception as e:
        # Catch any other unexpected errors during the API call or processing
        print(f"API test failed: An unexpected error occurred: {e}")

# Run the test function when the script is executed
if __name__ == "__main__":
    test_api()
