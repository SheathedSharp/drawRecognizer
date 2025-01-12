/*
 * @Author: SheathedSharp z404878860@163.com
 * @Date: 2025-01-12 14:55:57
 */
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

// 设置画笔样式
ctx.strokeStyle = "black";
ctx.lineWidth = 15;
ctx.lineCap = "round";

// 绘画事件监听
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

// 清除画布
document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "-";
});

// 识别按钮
document.getElementById("recognizeBtn").addEventListener("click", async () => {
  const imageData = canvas.toDataURL("image/png");
  try {
    const response = await fetch("/api/recognize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageData }),
    });
    const data = await response.json();
    document.getElementById("result").textContent = data.result;
  } catch (error) {
    console.error("Error:", error);
  }
});
