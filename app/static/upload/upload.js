document.addEventListener("DOMContentLoaded", function () {
    var successMessage = document.getElementById('success-message');
    if (successMessage) {
        setTimeout(function () {
            successMessage.style.display = 'none';
        }, 3000);
    }

    var selDiv = "";
    document.querySelector('#id_signature_files').addEventListener('change', ensureSingleFileSelected, false);
    selDiv = document.querySelector("#selectedFiles");

    function ensureSingleFileSelected(e) {
        if (!e.target.files) return;

        var files = e.target.files;
        if (files.length !== 1) {
            alert("Please select exactly one file");
            e.target.value = "";
        }
        else {
            handleFileSelect(e);
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
