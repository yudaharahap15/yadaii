function handleFormSubmission() {
    var form = document.forms['form1'];
    var formData = new FormData(form);

    fetch(form.action, {
        method: form.method,
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        alert(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}