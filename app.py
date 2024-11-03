from flask import Flask, request, jsonify
from pinecone import Pinecone
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Constants
EMBEDDING_DIMENSION = 1536  # text-embedding-ada-002 dimension
SCORE_THRESHOLD = 0.81

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('RAG_PINECONE_API_KEY'))
pinecone_index = pc.Index(os.environ.get('RAG_PINECONE_INDEX'))

def get_rag_context(query_embedding, score_thresh, top_k=5):
    retrieved_docs = pinecone_index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    context = ''
    for doc in retrieved_docs["matches"]:
        if doc["score"] > score_thresh:
            try:
                extracted_context = json.loads(doc["metadata"]["_node_content"])["metadata"]["text"]
                provenance = doc["metadata"]["c_document_id"]
            except Exception as e:
                extracted_context = doc["metadata"]["text"]
                provenance = doc["metadata"]["c_document_id"]
            
            context += f"{extracted_context}(Ref: {provenance}.)\n"
    
    if context:
        return context
    else:
        return 'Context not related to EDS'

def validate_embedding(embedding):
    """Validate the embedding vector format and dimension."""
    if not isinstance(embedding, list):
        return False, "query_embedding must be a list of numbers"
    
    if len(embedding) != EMBEDDING_DIMENSION:
        return False, f"query_embedding must have exactly {EMBEDDING_DIMENSION} dimensions (text-embedding-ada-002 format)"
    
    if not all(isinstance(x, (int, float)) for x in embedding):
        return False, "query_embedding must contain only numbers"
    
    return True, None

@app.route('/', methods=['GET'])
def home():
    """Root endpoint providing API documentation."""
    return jsonify({
        "name": "EDS Context Vector Database Search API",
        "description": """
        This API provides semantic search capabilities for rare disease Ehlers-Danlos Syndrome (EDS) related medical information. 
        It was specifically developed as part of the zebra-Llama project (https://huggingface.co/zebraLLAMA/zebra-Llama-v0.2), 
        an LLM model specialized in EDS-related queries.
        
        The vector database contains curated medical information about EDS from authoritative sources:
        - PubMed research papers
        - NCBI Gene Reviews
        
        The database includes comprehensive information about:
        - Clinical presentations and manifestations
        - Diagnostic criteria and guidelines
        - Treatment approaches and management strategies
        - Genetic and molecular aspects
        - Research findings and clinical studies
        
        While this API can be used with any LLM as an EDS context provider, it is recommended to be used with the zebra-Llama model for best results in EDS-related queries.
        
        The API returns relevant context passages from these verified sources, each with a reference citation 
        to the original PubMed paper or Gene Review.
        """,
        "version": "1.0",
        "data_sources": {
            "primary_sources": [
                "PubMed research papers",
                "NCBI Gene Reviews"
            ],
            "content_types": [
                "Clinical research papers",
                "Genetic studies",
                "Clinical guidelines",
                "Review articles",
                "Case studies"
            ]
        },
        "model_compatibility": {
            "embedding_model": "text-embedding-ada-002",
            "feature dimension": 1536
        },
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Returns API documentation and available endpoints"
            },
            {
                "path": "/search",
                "method": "POST",
                "description": "Performs semantic search for EDS-related context using provided embeddings",
                "request_body": {
                    "query_embedding": {
                        "type": "array",
                        "description": "1536-dimensional embedding from text-embedding-ada-002 model",
                        "required": True
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "required": False,
                        "default": 2
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "context": "String containing relevant EDS-related passages with source citations"
                        },
                    },
                    "400": {
                        "description": "Invalid request (wrong embedding format/dimension)",
                        "content": {
                            "error": "error type",
                            "details": "detailed error message"
                        }
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "error": "error type",
                            "details": "error message"
                        }
                    }
                }
            }
        ]
    })

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get data from request
        data = request.get_json()
        
        # Check if query_embedding exists
        if 'query_embedding' not in data:
            return jsonify({
                'error': 'query_embedding is required',
                'details': 'Must provide an embedding from text-embedding-ada-002 model'
            }), 400
        
        # Validate embedding
        query_embedding = data['query_embedding']
        is_valid, error_message = validate_embedding(query_embedding)
        if not is_valid:
            return jsonify({
                'error': 'Invalid embedding format',
                'details': error_message
            }), 400
        
        # Validate top_k
        top_k = data.get('top_k', 2)
        if not isinstance(top_k, int) or top_k < 1:
            return jsonify({
                'error': 'Invalid top_k value',
                'details': 'top_k must be a positive integer'
            }), 400
        
        # Get context
        context = get_rag_context(query_embedding, SCORE_THRESHOLD, top_k=top_k)
        
        # Return response
        return jsonify({'context': context})
    
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON',
            'details': 'Request body must be valid JSON'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    print("Running in development mode with Flask's built-in server")
    app.run(debug=True, port=5000)
