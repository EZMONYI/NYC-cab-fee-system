from google.cloud import automl


class AutoMLEngineClient:
    """
    AutoMLEngineClient recognize specific NYC restaurants images and their map coordinates
    according to your custom ML model.
    You should NOT change this class.

    Methods:
        get_prediction(self, content):
    """

    def __init__(self, project_id, model_id):
        """
       Constructs all the necessary attributes for the AutoMLEngineClient object

       Parameters:
           project_id (str): project ID of your GCP project
           model_id (str): ID of your AutoML model
       """
        self.prediction_client = automl.PredictionServiceClient()
        self.name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)

    def get_prediction(self, content):
        """
        Use the custom AutoML model to identify specific NYC restaurants and their map coordinates


        Arguments:
            content(str): byte stream of a specific NYC restaurant image

        Returns:
            an object that contains label and coordinates of the restaurant
        """
        payload = {'image': {'image_bytes': content}}
        params = {}
        request = automl.PredictRequest(name=self.name, payload=payload, params=params)
        response = self.prediction_client.predict(request)
        return response  # waits till request is returned
