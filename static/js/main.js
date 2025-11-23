document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("leafImage");
  const previewImage = document.getElementById("previewImage");
  const previewPlaceholder = document.getElementById("previewPlaceholder");
  const form = document.getElementById("uploadForm");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const spinner = document.getElementById("loadingSpinner");
  const resultCard = document.getElementById("resultCard");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultBadge = document.getElementById("resultBadge");
  const resultConfidence = document.getElementById("resultConfidence");
  const resultDescription = document.getElementById("resultDescription");
  const resultTimestamp = document.getElementById("resultTimestamp");
  const resultImage = document.getElementById("resultImage");
  const resultError = document.getElementById("resultError");

  if (!fileInput || !form) return;

  const setLoading = (state) => {
    if (state) {
      analyzeBtn.setAttribute("disabled", "disabled");
      spinner.classList.remove("d-none");
    } else {
      analyzeBtn.removeAttribute("disabled");
      spinner.classList.add("d-none");
    }
  };

  const resetResult = () => {
    resultCard?.classList.add("d-none");
    resultEmpty?.classList.remove("d-none");
    resultError?.classList.add("d-none");
  };

  fileInput.addEventListener("change", () => {
    resetResult();
    const file = fileInput.files[0];
    if (!file) {
      previewImage.classList.add("d-none");
      previewPlaceholder.classList.remove("d-none");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("d-none");
      previewPlaceholder.classList.add("d-none");
    };
    reader.readAsDataURL(file);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultError?.classList.add("d-none");
    const file = fileInput.files[0];
    if (!file) {
      resultError.textContent = "Please choose an image first.";
      resultError.classList.remove("d-none");
      return;
    }
    const formData = new FormData();
    formData.append("image", file);
    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }
      resultBadge.textContent = data.prediction;
      resultBadge.className = `badge ${data.prediction === "Healthy" ? "bg-success-subtle text-success" : "bg-danger-subtle text-danger"}`;
      resultConfidence.textContent = `${data.confidence.toFixed(2)}%`;
      resultDescription.textContent = data.description;
      resultTimestamp.textContent = data.timestamp;
      resultImage.src = data.image_path;
      resultCard.classList.remove("d-none");
      resultEmpty.classList.add("d-none");
    } catch (error) {
      resultError.textContent = error.message;
      resultError.classList.remove("d-none");
    } finally {
      setLoading(false);
    }
  });
});

