document.addEventListener("DOMContentLoaded", function () {
    var successMessage = document.getElementById('success-message');
    if (successMessage) {
        setTimeout(function () {
            successMessage.style.display = 'none';
        }, 3000);
    }

    var selDiv = "";
    document.querySelector('#id_signature_files').addEventListener('change', ensureTwentyFiveFilesSelected, false);
    selDiv = document.querySelector("#selectedFiles");

    function ensureTwentyFiveFilesSelected(e) {
        if (!e.target.files) return;

        var files = e.target.files;
        if (files.length < 25) {
            alert("Please select twenty-five images");
            e.target.value = "";
        } else if (files.length > 25) {
            alert("Please select only twenty-five images");
            e.target.value = "";
        }
    }

    function handleFileSelect(e) {
        if (!e.target.files) return;

        selDiv.innerHTML = "";

        var files = e.target.files;
        for (var i = 0; i < files.length; i++) {
            var f = files[i];

            selDiv.innerHTML += f.name + "<br/>";
        }
    }
});
