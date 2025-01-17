/*
 * @Author: SheathedSharp z404878860@163.com
 * @Date: 2025-01-12 14:56:21
 */
const canvas = document.getElementById("sketchCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

ctx.strokeStyle = "black";
ctx.lineWidth = 3;
ctx.lineCap = "round";

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  draw(e);
}

function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function stopDrawing() {
  isDrawing = false;
  ctx.beginPath();
}

document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("results").innerHTML = "";
});

document.getElementById("searchBtn").addEventListener("click", async () => {
  const imageData = canvas.toDataURL("image/png");
  try {
    const response = await fetch("/api/retrieve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sketch: imageData }),
    });
    const data = await response.json();
    displayResults(data.results);
  } catch (error) {
    console.error("Error:", error);
  }
});

function displayResults(results) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = '';
  
  results.forEach(result => {
      const col = document.createElement("div");
      col.className = "col-md-4";
      
      const img = document.createElement("img");
      img.src = result.path;
      img.className = "img-fluid";
      
      const similarity = document.createElement("p");
      similarity.textContent = `Similarity: ${result.similarity}`;
      
      col.appendChild(img);
      col.appendChild(similarity);
      resultsDiv.appendChild(col);
  });
}
