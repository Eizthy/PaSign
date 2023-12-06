(function($) {
    $(document).ready(function() {
        $("#loginButton").on("click", function() {
            // Get the values of username and password
            var username = $("input[name='username']").val();
            var password = $("input[name='password']").val();

            // Check if either username or password is empty
            if (!username || !password) {
                // Display an alert or handle the error as needed
                alert("Please enter both username and password.");
            } else {
                // Proceed with form submission
                window.location.href = "{% url 'dashboard_page' %}";
            }
        });
    });
})(jQuery);