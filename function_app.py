import azure.functions as func
import json
import logging
import os
from transformers import AutoTokenizer
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="TokenCounter", methods=[func.HttpMethod.POST])
def TokenCounter(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handle TokenCounter POST requests by counting request and response tokens.
    
    This function extracts the request and response bodies from the incoming request,
    tokenizes them using the Llama tokenizer, and returns token counts.
    
    Args:
        req: The HTTP request containing RequestBody and ResponseBody fields
        
    Returns:
        HTTP response with token usage metrics (prompt_tokens, completion_tokens, total_tokens)
    """
    logging.info("Python HTTP trigger function processed a request.")
    
    # Authenticate with Hugging Face
    login(token=os.environ.get("HF_TOKEN", ""))

    # Initialize tokenizer for Llama model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    try:
        # Parse incoming request body
        request_body = req.get_json()
        logger.info(f"Received request body: {request_body}")

        # Extract request and response body strings
        request_body_json_str = str(request_body.get("RequestBody", ""))
        response_body_json_str = str(request_body.get("ResponseBody", ""))

        # Parse JSON strings into objects
        request_json = json.loads(request_body_json_str)
        response_json = json.loads(response_body_json_str)
        logger.info(f"Parsed request JSON: {request_json}")

        # Extract completion message content from response
        # Navigate: response -> choices[0] -> message -> content
        choices = response_json.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message_obj = first_choice.get("message", {})
        completion_message = message_obj.get("content", "")
        
        logger.info(f"Extracted completion message: {completion_message}")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return func.HttpResponse(
            json.dumps(
                {
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    }
                }
            ),
            status_code=200,
        )

    # Tokenize request and response
    request_tokens = tokenizer.tokenize(request_body_json_str)
    completion_tokens_list = tokenizer.tokenize(completion_message)
    
    logger.debug(f"Request tokens: {request_tokens}")
    logger.debug(f"Completion tokens: {completion_tokens_list}")

    # Count tokens in request and completion
    prompt_tokens = len(request_tokens)
    completion_tokens = len(completion_tokens_list)
    total_tokens = prompt_tokens + completion_tokens

    # Prepare response with token counts
    response_data = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    }

    logger.info(f"Token counts: {response_data}")

    return func.HttpResponse(
        json.dumps(response_data),
        status_code=200,
    )
