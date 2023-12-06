document.getElementById('imgInp4').onchange = evt => {
    const fileInput = evt.target;
    const [file] = fileInput.files;

    if (file) {
        const previewImage = document.getElementById('blah4');
        previewImage.src = URL.createObjectURL(file);
    }
};

// Event listener for student ID input
document.getElementById('studentIdInput').addEventListener('input', function () {
    let studentId = this.value.trim(); // Trim any leading or trailing whitespaces

    // Remove existing hyphens
    studentId = studentId.replace(/-/g, '');

    // Insert hyphens at the correct positions
    if (studentId.length >= 4) {
        studentId = studentId.substring(0, 4) + '-' + studentId.substring(4, 5) + '-' + studentId.substring(5, 9);
    }

    // Update the input field with the formatted studentId
    this.value = studentId;

    if (studentId.length === 11) { // Check if the full pattern is matched
        // Make an AJAX request to the server to fetch corresponding name, email, and signature file URL
        fetch(`/get-student-data/${encodeURIComponent(studentId)}/`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(data); // Log the data to the console for debugging

                document.getElementById('nameInput').value = data.name || '';
                document.getElementById('emailInput').value = data.email || '';

                const imagePreview = document.getElementById('blah1');
                const signatureFieldKey = 'signature_1_url'; // Assuming only one signature field

                if (data[signatureFieldKey]) {
                    // Do not include '/media/' again; it's already in the data[signatureFieldKey]
                    imagePreview.src = data[signatureFieldKey];
                    imagePreview.style.display = 'block'; // Show the image preview element
                } else {
                    imagePreview.src = ''; // Clear the image preview element
                    imagePreview.style.display = 'none'; // Hide the image preview element
                }
            })
            .catch(error => {
                console.error('Error fetching student data:', error);
                // Handle the error, e.g., display a message to the user
            });
    } else {
        // Clear name, email, and image preview if the pattern is not fully matched
        document.getElementById('nameInput').value = '';
        document.getElementById('emailInput').value = '';
        const imagePreview = document.getElementById('blah1');
        imagePreview.src = ''; // Clear the image preview element
        imagePreview.style.display = 'none'; // Hide the image preview element
    }
});
