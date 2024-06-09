let webcamStream;
let videoElement = document.getElementById('webcam-preview');
let predictionCanvas = document.getElementById('predictionCanvas');
let predictionContext = predictionCanvas.getContext('2d');

function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            videoElement.srcObject = stream;
            webcamStream = stream;
        })
        .catch(function (error) {
            console.error('Error accessing webcam:', error);
        });
}

function startRecording() {
    // Add logic to start recording if needed
    setInterval(updatePrediction, 1000); // Update prediction every second (adjust as needed)
}

function stopRecording() {
    // Add logic to stop recording if needed
}

function takeSnapshot() {
    // Add logic to take a snapshot from the webcam and save it to the server
    fetch('/take_snapshot')
        .then(response => response.json())
        .then(data => {
            console.log('Snapshot taken:', data);
        });
}

function updatePrediction() {
    // Fetch prediction result from server and draw on canvas
    fetch('/video_feed')
        .then(response => response.blob())
        .then(blob => {
            let image = new Image();
            image.onload = function () {
                predictionContext.drawImage(image, 0, 0, predictionCanvas.width, predictionCanvas.height);
            };
            image.src = URL.createObjectURL(blob);
        });
}
