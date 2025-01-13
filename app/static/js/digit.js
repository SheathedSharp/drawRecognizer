/*
 * @Author: SheathedSharp z404878860@163.com
 * @Date: 2025-01-12 14:55:57
 */
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

// 设置画布背景为白色
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

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
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
}

function stopDrawing() {
  isDrawing = false;
}

// 清除画布
document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").textContent = "-";
});

// 识别按钮
document.getElementById("recognizeBtn").addEventListener("click", async () => {
  // 获取画布数据前确保所有绘制操作已完成
  ctx.closePath();

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

// 获取模型信息并填充选择器和性能表格
async function loadModelInfo() {
  try {
    const response = await fetch("/api/models");
    const models = await response.json();

    const modelSelect = document.getElementById("modelSelect");
    const modelStats = document.getElementById("modelStats");

    // 填充模型选择器
    Object.entries(models).forEach(([name, info]) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = info.description;
      modelSelect.appendChild(option);
    });

    // 填充性能对比表格
    Object.entries(models).forEach(([name, info]) => {
      const stats = info.stats || {};
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${info.description}</td>
        <td>${info.type}</td>
        <td>${stats.best_accuracy ? stats.best_accuracy.toFixed(2) : "N/A"}%</td>
        <td>${info.parameters.toLocaleString()}</td>
        <td>${stats.training_time ? stats.training_time.toFixed(2) : "N/A"}s</td>
      `;
      modelStats.appendChild(row);
    });
  } catch (error) {
    console.error("Error loading model info:", error);
  }
}

// 切换模型
async function switchModel(modelName) {
  try {
    const response = await fetch("/api/switch_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_name: modelName }),
    });
    const result = await response.json();
    if (response.ok) {
      console.log(result.message);
    } else {
      console.error(result.error);
    }
  } catch (error) {
    console.error("Error switching model:", error);
  }
}

// 页面加载完成后初始化
document.addEventListener("DOMContentLoaded", () => {
  loadModelInfo();

  // 监听模型选择变化
  document.getElementById("modelSelect").addEventListener("change", (e) => {
    switchModel(e.target.value);
  });
});
