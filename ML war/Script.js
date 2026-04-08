async function uploadVideo() {

    let file = document.getElementById("videoUpload").files[0]

    let formData = new FormData()

    formData.append("video", file)


    let response = await fetch("http://localhost:8000/extract_frames", {

        method: "POST",
        body: formData

    })

    let frames = await response.json()

    displayFrames(frames)

}



function displayFrames(frames) {

    let container = document.getElementById("frameContainer")

    container.innerHTML = ""

    frames.forEach(frame => {

        let img = document.createElement("img")

        img.src = frame

        container.appendChild(img)

    })

}



async function predictOrder() {

    let response = await fetch("http://localhost:8000/predict")

    let result = await response.json()

    document.getElementById("predictionResult").innerText =
        "Predicted Order: " + result.order

}