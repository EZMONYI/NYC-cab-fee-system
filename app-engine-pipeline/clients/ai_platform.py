import googleapiclient.discovery


class AIPlatformClient:
    """
    Make predictions using the model you uploaded to AI platform
    You should NOT change this class.

    Methods:
        predict(self, instances): make predictions using the model and return predicted fare
    """

    def __init__(self, project, model, version):
        """
        Constructs all the necessary attributes for the AIPlatformClient object

        Parameters:
            project (str): project ID of your GCP project
            model (str): name of the model you are using
            version (str): the model version of the model you are using
        """
        self.ml_service = googleapiclient.discovery.build('ml', 'v1')
        self.name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    def predict(self, instances):
        """
        Make predictions using the model you uploaded to AI platform

        Arguments:
            instances: a DataFrame with the predictors created

        Returns:
            predicted fare
        """
        response = self.ml_service.projects().predict(
            name=self.name,
            body={'instances': instances}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']
