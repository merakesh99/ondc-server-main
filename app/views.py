from django.shortcuts import render

# Create your views here.
# def predict(request):

from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import cv2
import keras_ocr
from django.http import JsonResponse
from django.shortcuts import render, redirect


def index(request):
    return render(request, 'app/pest.html')









pipeline = keras_ocr.pipeline.Pipeline()


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image = request.FILES['image'].read()

            try:
                # Convert the image bytes to a format compatible with cv2
                image_np = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

                # Perform OCR on the image
                prediction_groups = pipeline.recognize([image_np])

                # Extract text from the predictions
                extracted_text = [word[0] for group in prediction_groups for word in group]

                # Display the results (optional)


                # Return the extracted text in the response
                response_data = {'prediction': extracted_text}
                return JsonResponse(response_data)
            except Exception as e:
                return JsonResponse({'error': f'Error processing the image: {str(e)}'}, status=500)
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
