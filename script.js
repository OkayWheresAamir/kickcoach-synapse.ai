const videoUpload = document.getElementById("videoUpload");
const videoPreview = document.getElementById("videoPreview");
const uploadIcon = document.getElementById("uploadIcon");

videoUpload.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      videoPreview.src = e.target.result;
      videoPreview.classList.remove("hidden");
      uploadIcon.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }
});
