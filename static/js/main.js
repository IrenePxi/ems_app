// Global utility functions for Daily EMS Sandbox

window.EMSApp = {
    showAlert: function(message, type) {
        const alert = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>`;
        $('main').prepend(alert);
        setTimeout(function() {
            $('.alert').fadeOut();
        }, 3000);
    },
    
    formatNumber: function(num, decimals = 2) {
        return num.toFixed(decimals);
    },
    
    formatPercent: function(num) {
        return (num * 100).toFixed(1) + '%';
    },
    
    formatDate: function(date) {
        return new Date(date).toLocaleDateString();
    },
    
    formatTime: function(date) {
        return new Date(date).toLocaleTimeString();
    }
};

// Global AJAX error handler
$(document).ajaxError(function(event, xhr) {
    if (xhr.status === 401 || xhr.status === 403) {
        window.location.href = '/';
    } else if (xhr.status >= 500) {
        EMSApp.showAlert('Server error. Please try again later.', 'danger');
    }
});

// Initialize Bootstrap tooltips and popovers
$(document).ready(function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});
