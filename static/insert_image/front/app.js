const imgInput = document.getElementById("imgInput");
const orig = document.getElementById("orig");
const lime = document.getElementById("lime");
const riskText = document.getElementById("riskText");
const analyzeBtn = document.getElementById("analyzeBtn");

let selectedFile = null;

imgInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    orig.src = URL.createObjectURL(selectedFile);
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("이미지를 선택해줘!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    riskText.textContent = `${(data.collision_prob * 100).toFixed(1)}%`;

    const filename = data.lime_image_path.split("\\").pop().split("/").pop();
    lime.src = `http://localhost:8000/lime/${filename}`;
});
