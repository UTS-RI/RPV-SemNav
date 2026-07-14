# Adam Scicluna, 2026, for RPV pipeline. Based on blip2itm.py.

from typing import Any, List, Optional

import numpy as np
import torch
import os

# CLIP imports for setup and processing
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
cossim = CosineSimilarity(dim=0, eps=1e-6)  # Cosine Similarity function for computing similarity between embeddings

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


class CLIP:
    """
    Initialise CLIP model
    """

    def __init__(
        self,
        model_type: str = "openai/clip-vit-base-patch32",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load model, tokenizer, text encoder, using the transformers library for embeddings length 77,512
        self.tokenizer = CLIPTokenizer.from_pretrained(model_type)
        self.text_encoder = CLIPTextModel.from_pretrained(model_type).to(device)
        self.model = CLIPModel.from_pretrained(model_type).to(device)
        self.device = device

        # Boolean flag to indicate whether to run direct object-object signal or object-room-object RPV
        self._direct_object_object: bool = os.environ.get("DIRECT_OBJECT_OBJECT", "false").lower() in ("true", "1", "yes") # Default to RPV mode
        print(f"SCICLUNA CLIP: Direct object-object mode: {self._direct_object_object}")

    def compute_text_embeddings(self, texts: Any) -> torch.Tensor:
        """
        Compute the text embeddings for a given text prompt.

        Args:
            text (str): The input text prompt.
        Returns:
            torch.Tensor: The text embeddings for the input text prompt.
        """
        # Convert single string to list if necessary
        if not isinstance(texts, list):
            texts = [texts]

        with torch.no_grad():
            text_inputs = self.tokenizer(
                texts, 
                padding="max_length", 
                return_tensors="pt").to(self.device)
            text_embeddings = torch.flatten(self.text_encoder(text_inputs.input_ids)['last_hidden_state'],1,-1)
        
        return text_embeddings


    def cosine_vector(self, query: str or List[str], candidates: List[str]) -> torch.Tensor:
        """
        Compute cosine similarity between a query string (or list of strings) and a list of candidate strings.
        Returns tensor of shape [NxM] where N is the number of queries and M is the number of candidates.
        """
        #print(f"SCICLUNA CLIP: Computing cosine similarity between query '{query}' and candidates {candidates}")
        q = self.compute_text_embeddings(query)  # [N, D] where N is 1 if query is a single string, or more if it's a list of strings
        #print(f'Shape of query embedding (should be N x D): {q.shape}')
        c = self.compute_text_embeddings(candidates)  # [M, D] where M is the number of candidates, D is the embedding dimension
        #print(f'Shape of candidate embeddings (should be M x D): {c.shape}')

        #print(f"SCICLUNA CLIP Cosine_Vector: Query embeddings shape: {q.shape}")
        #print(f"SCICLUNA CLIP Cosine_Vector: Candidate embeddings shape: {c.shape}")

        sims = torch.nn.functional.cosine_similarity(q.unsqueeze(1), c.unsqueeze(0), dim=-1)  # [N, M] where each entry (i,j) is the cosine similarity between query i and candidate j
        #print(f'Shape of similarity vector (should be N x M): {sims.shape}')
        return sims
        


    def apply_softmax(self, similarities: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
        """
        Apply temperature-scaled softmax to similarity vector to obtain probabilities.

        Tensor is 1xN
        """
        probabilities = torch.nn.functional.softmax(similarities / temperature, dim=-1)
        #print(f"SCICLUNA CLIP: Applying softmax to similarity vector: \nSimilarities: {similarities} \nProbabilities: {probabilities}")
        # Check that rows (probability distributions) sum to 1
        #row_sums = probabilities.sum(dim=-1)
        #print(f"Row sums (should be 1): {row_sums}")

        return probabilities


    def calculate_dot_product(self, rpv1: torch.Tensor, rpv_matrix: torch.Tensor) -> torch.Tensor:
        """
        Dot product between a probability vector (rpv1: [1, M]) and a matrix (rpv_matrix: [N, M]) 
        → [1, N] (dot product for each target RPV probability distribution against the detection RPVs)
        """
        # Return one dot product per detection (shape [N]) so downstream len matches number of segments
        #print(f"SCICLUNA CLIP: Calculating dot product between target probability vector and matrix of detection RPVs.")
        #print(f"Target probability vector (rpv1): {rpv1}, shape: {rpv1.shape}")
        #print(f"Matrix of detection RPVs (rpv_matrix): {rpv_matrix}, shape: {rpv_matrix.shape}")
        dot_product = torch.matmul(rpv_matrix, rpv1.T).squeeze(-1)
        #print(f"Dot product result (one per detection): {dot_product}, shape: {dot_product.shape}")
        return dot_product


class CLIPClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/clip"

    def cooccurrence(
        self,
        query: str or List[str],
        candidates: List[str],
        temperature: float = 0.03,
        target_prob_dist: Optional[List[float]] = None,
    ) -> dict:
        """
        Request similarity vector (and optional dot products) from the CLIP server.
        Returns a dict with keys: similarities, probabilities, and optionally dot_products.
        """
        payload = {
            "query": query,
            "candidates": candidates,
            "temperature": float(temperature),
        }
        if target_prob_dist is not None:
            payload["target_prob_dist"] = target_prob_dist

        response = send_request(self.url, **payload)
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class CLIPServer(ServerMixin, CLIP):
        def process_payload(self, payload: dict) -> dict:
            # Extract query, candidates, and temperature from the payload
            query = payload["query"]
            candidates = payload["candidates"]
            temperature = float(payload.get("temperature", 0.03))

            #print(f"\nCLIP SERVER: Received request with query: {query}, candidates: {candidates}, temperature: {temperature}")

            # Calculate cosine similarity between query and candidates
            # ISSUE: What if there are multiple queries?
            sims = self.cosine_vector(query, candidates)

            print(f"SCICLUNA CLIP SERVER: Similarity vector: {sims}, Shape: {sims.shape}")
            print(f"SCICLUNA CLIP SERVER: Candidates (detections): {candidates}, Target: {query}")

            # Apply softmax over similarity vector to get discrete probability distribution
            if not self._direct_object_object:
                #print(f"SCICLUNA CLIP SERVER: RPV mode")
                probs = self.apply_softmax(sims, temperature)
            
                # Calculate the dot product between a target probability distributions and the detected object probability distributions
                if "target_prob_dist" in payload:
                    # Should be 1xM vector where M is the number of candidates (same length as sims and probs)
                    target_prob_dist = torch.tensor(payload["target_prob_dist"], device=self.device)
                    
                    # Detections probability distributions should be shape NxM where N is the number of detected objects and M is the number of candidates (same length as sims and probs)
                    # Target probability distribution should be shape 1xM where M is the number of candidates (same length as sims and probs)
                    # Dot product vector should then be (NxM) x (Mx1) → Nx1, where N is the number of detected objects and the single value for each object is the dot product between the target probability distribution and the object's probability distribution over the candidate labels.
                    dot_products = self.calculate_dot_product(target_prob_dist, probs)



                    response = {
                        "similarities": sims.tolist(),
                        "probabilities": probs.tolist(),
                        "dot_products": dot_products.tolist(),
                    }
                else:
                    print(f"Shape of probs: {probs.shape}, Shape of sims: {sims.shape}")

                    response = {
                        "similarities": sims.tolist(),
                        "probabilities": probs.tolist(),
                    }

            else:
                # If direct object-object mode, just return similarities without softmax or dot products
                # Make sure dimension is (,M) so returned list is 1D
                assert sims.dim() == 2 and sims.size(0) == 1, f"Expected similarity vector to be of shape (1, M), but got {sims.shape}"
                sims = sims.squeeze(0)
                #print(f"Sims shape after squeeze: {sims.shape}, should be 1D with length equal to number of candidates.")
                response = {
                    "similarities": sims.tolist(),
                }

            return response


    clip = CLIPServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(clip, name="clip", port=args.port)
