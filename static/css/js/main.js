document.addEventListener('DOMContentLoaded', function() {
    console.log('main.js loaded'); // Debug: Confirm script runs

    // Close flash messages
    document.querySelectorAll('.flash-close').forEach(button => {
        button.addEventListener('click', function() {
            this.parentElement.style.opacity = '0';
            setTimeout(() => this.parentElement.remove(), 300);
        });
    });

    // Auto-hide flash messages after 5 seconds
    setTimeout(() => {
        document.querySelectorAll('.flash-message').forEach(message => {
            message.style.opacity = '0';
            setTimeout(() => message.remove(), 300);
        });
    }, 5000);

    // Initialize date inputs
    const dateInputs = document.querySelectorAll('input[type="date"]');
    const today = new Date().toISOString().split('T')[0];
    const currentPath = window.location.pathname;
    console.log('Current path:', currentPath); // Debug path
    dateInputs.forEach(input => {
        if (!input.value) {
            input.value = today;
        }
        // Set min only for non-historical pages
        if (!currentPath.startsWith('/historical')) {
            input.min = today;
        } else {
            input.removeAttribute('min');
        }
    });

    // Add validation between start and end dates on historical page
    if (currentPath.startsWith('/historical')) {
        const startDateInput = document.getElementById('start_date');
        const endDateInput = document.getElementById('end_date');

        if (startDateInput && endDateInput) {
            startDateInput.addEventListener('change', function() {
                // When start date changes, update end date min to be >= start date
                endDateInput.min = this.value;
                
                // If current end date is before new start date, reset it
                if (endDateInput.value && endDateInput.value < this.value) {
                    endDateInput.value = this.value;
                }
            });

            endDateInput.addEventListener('change', function() {
                // When end date changes, validate it's not before start date
                if (startDateInput.value && this.value < startDateInput.value) {
                    alert('End date must be on or after start date');
                    this.value = startDateInput.value;
                }
            });
        }
    }

    // Initialize location autocomplete only on index page
    const locationInput = document.getElementById('location');
    const suggestionsDropdown = document.getElementById('location-suggestions');

    if (currentPath === '/' || currentPath === '/index') {
        if (!locationInput || !suggestionsDropdown) {
            console.error('DOM elements missing: locationInput=', !!locationInput, 'suggestionsDropdown=', !!suggestionsDropdown);
            return;
        }
        console.log('Autocomplete elements found'); // Debug

        locationInput.addEventListener('input', async () => {
            const query = locationInput.value.trim();
            console.log('Input query:', query); // Debug

            if (query === '') {
                suggestionsDropdown.innerHTML = '';
                suggestionsDropdown.style.display = 'none';
                return;
            }

            try {
                const configResponse= await fetch('/config');
                const config = await configResponse.json();
                console.log('Fetching suggestions for:', query); // Debug
                const response = await fetch(`https://api.geoapify.com/v1/geocode/autocomplete?text=${encodeURIComponent(query)}&apiKey=${config.geoapifyApiKey}`, {
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                console.log('API response status:', response.status); // Debug
                if (!response.ok) {
                    throw new Error(`API request failed with status ${response.status}`);
                }

                const data = await response.json();
                console.log('API response data:', data); // Debug
                displayResults(data.features || [], query.length);
            } catch (error) {
                console.error('Fetch error:', error); // Debug
                suggestionsDropdown.innerHTML = '<div class="suggestion-item" style="color: red;">Error fetching suggestions</div>';
                suggestionsDropdown.style.display = 'block';
            }
        });

        function displayResults(features, queryLength) {
            suggestionsDropdown.innerHTML = '';
            console.log('Displaying features:', features); // Debug

            if (!features || features.length === 0) {
                if (queryLength === 1) {
                    suggestionsDropdown.innerHTML = '<div class="suggestion-item">Please enter 2 or more letters for suggestions</div>';
                } else {
                    suggestionsDropdown.innerHTML = '<div class="suggestion-item">No suggestions found</div>';
                }
                suggestionsDropdown.style.display = 'block';
                return;
            }

            features.forEach(feature => {
                const city = feature.properties.city || feature.properties.name || feature.properties.formatted;
                const displayText = feature.properties.city && feature.properties.country
                    ? `${feature.properties.city}, ${feature.properties.country}`
                    : feature.properties.formatted;

                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.textContent = displayText;
                item.dataset.city = city; // Store city for backend
                item.addEventListener('click', () => {
                    console.log('Selected city:', city); // Debug
                    locationInput.value = city; // Send only city name to backend
                    suggestionsDropdown.innerHTML = '';
                    suggestionsDropdown.style.display = 'none';
                });
                suggestionsDropdown.appendChild(item);
            });
            suggestionsDropdown.style.display = 'block';
        }

        // Hide dropdown when clicking outside
        document.addEventListener('click', function(event) {
            if (!locationInput.contains(event.target) && !suggestionsDropdown.contains(event.target)) {
                console.log('Hiding dropdown'); // Debug
                suggestionsDropdown.style.display = 'none';
            }
        });
    } else {
        console.log('Autocomplete skipped on path:', currentPath); // Debug
    }

    // Test dropdown visibility (only on index page)
    if (currentPath === '/' || currentPath === '/index') {
        console.log('Testing dropdown'); // Debug
        suggestionsDropdown.innerHTML = '<div class="suggestion-item">Test: Autocomplete Loaded</div>';
        suggestionsDropdown.style.display = 'block';
        setTimeout(() => {
            suggestionsDropdown.style.display = 'none';
        }, 1000);
    }
});