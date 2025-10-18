/**
 * OpenStreetMap Location Picker
 * Provides interactive map for location selection
 */

class LocationPicker {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.map = null;
        this.marker = null;
        this.latitude = options.latitude || null;
        this.longitude = options.longitude || null;
        this.zoom = options.zoom || 12;
        this.center = options.center || [13.4217, 123.4175]; // Default to Iriga City, Rinconada area
        
        // Define boundaries for Rinconada area in Camarines Sur
        this.bounds = [
            [13.2, 123.2],  // Southwest corner
            [13.6, 123.7]   // Northeast corner
        ];
        
        // Initialize the map
        this.initMap();
    }

    initMap() {
        // Check if Leaflet is loaded
        if (typeof L === 'undefined') {
            console.error('Leaflet library not loaded. Please check if Leaflet CSS and JS are included.');
            return;
        }

        // Check if container exists
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Map container with id '${this.containerId}' not found.`);
            return;
        }

        // Set initial center if coordinates are provided
        if (this.latitude && this.longitude) {
            this.center = [this.latitude, this.longitude];
        }

        console.log('Initializing map with center:', this.center, 'zoom:', this.zoom);

        // Initialize the map
        this.map = L.map(this.containerId, {
            maxBounds: this.bounds,
            maxBoundsViscosity: 1.0
        }).setView(this.center, this.zoom);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 16
        }).addTo(this.map);
        
        // Restrict map to Rinconada area bounds
        this.map.setMaxBounds(this.bounds);
        
        console.log('Map tiles added successfully - Restricted to Rinconada area');

        // Add marker if coordinates are provided
        if (this.latitude && this.longitude) {
            this.addMarker(this.latitude, this.longitude);
        }

        // Add click event to place marker
        this.map.on('click', (e) => {
            // Check if click is within Rinconada area bounds
            if (this.isWithinBounds(e.latlng.lat, e.latlng.lng)) {
                this.addMarker(e.latlng.lat, e.latlng.lng);
                this.updateFormFields(e.latlng.lat, e.latlng.lng);
            } else {
                alert('Please select a location within the Rinconada area in Camarines Sur.');
            }
        });

        // Add geocoding control
        this.addGeocodingControl();
    }

    addMarker(lat, lng) {
        // Remove existing marker
        if (this.marker) {
            this.map.removeLayer(this.marker);
        }

        // Add new marker
        this.marker = L.marker([lat, lng]).addTo(this.map);
        this.latitude = lat;
        this.longitude = lng;

        // Add popup with coordinates
        this.marker.bindPopup(`
            <div class="text-center">
                <strong>Selected Location</strong><br>
                <small>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}</small>
            </div>
        `).openPopup();
    }

    addGeocodingControl() {
        try {
            // Create geocoding control
            const geocodingControl = L.Control.extend({
                onAdd: function(map) {
                    const div = L.DomUtil.create('div', 'leaflet-control-geocoding');
                    div.innerHTML = `
                        <div class="input-group">
                            <input type="text" class="form-control" id="geocodingInput" 
                                   placeholder="Search for a location...">
                            <button class="btn btn-outline-secondary" type="button" id="geocodingButton">
                                <i class="bi bi-search"></i>
                            </button>
                        </div>
                    `;
                    
                    // Add event listeners
                    const input = div.querySelector('#geocodingInput');
                    const button = div.querySelector('#geocodingButton');
                    
                    const searchLocation = () => {
                        const query = input.value.trim();
                        if (query) {
                            this.geocodeLocation(query);
                        }
                    };
                    
                    button.addEventListener('click', searchLocation);
                    input.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            searchLocation();
                        }
                    });
                    
                    return div;
                }
            });

            // Add the control to the map
            this.map.addControl(new geocodingControl());
            console.log('Geocoding control added successfully');
        } catch (error) {
            console.error('Error adding geocoding control:', error);
        }
    }

    async geocodeLocation(query) {
        try {
            // Use Nominatim geocoding service with Rinconada area restriction
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&bounded=1&viewbox=123.2,13.2,123.7,13.6`
            );
            const data = await response.json();
            
            if (data && data.length > 0) {
                // Find the first result within Rinconada bounds
                let validResult = null;
                for (const result of data) {
                    const lat = parseFloat(result.lat);
                    const lng = parseFloat(result.lon);
                    if (this.isWithinBounds(lat, lng)) {
                        validResult = result;
                        break;
                    }
                }
                
                if (validResult) {
                    const lat = parseFloat(validResult.lat);
                    const lng = parseFloat(validResult.lon);
                    
                    // Update map view and marker
                    this.map.setView([lat, lng], 15);
                    this.addMarker(lat, lng);
                    this.updateFormFields(lat, lng);
                    
                    // Update the search input with the found address
                    document.getElementById('geocodingInput').value = validResult.display_name;
                } else {
                    alert('No locations found within the Rinconada area. Please search for a location in Iriga City, Baao, Buhi, Nabua, or Bato.');
                }
            } else {
                alert('Location not found. Please search for a location within the Rinconada area in Camarines Sur.');
            }
        } catch (error) {
            console.error('Geocoding error:', error);
            alert('Error searching for location. Please try again.');
        }
    }

    updateFormFields(lat, lng) {
        // Update hidden form fields
        const latField = document.getElementById('latitude');
        const lngField = document.getElementById('longitude');
        
        if (latField) latField.value = lat;
        if (lngField) lngField.value = lng;

        // Update location text field with reverse geocoded address
        this.reverseGeocode(lat, lng);
    }

    async reverseGeocode(lat, lng) {
        try {
            const response = await fetch(
                `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`
            );
            const data = await response.json();
            
            if (data && data.display_name) {
                const locationField = document.getElementById('location');
                if (locationField) {
                    locationField.value = data.display_name;
                }
            }
        } catch (error) {
            console.error('Reverse geocoding error:', error);
        }
    }

    setLocation(lat, lng) {
        if (this.isWithinBounds(lat, lng)) {
            this.latitude = lat;
            this.longitude = lng;
            this.map.setView([lat, lng], this.zoom);
            this.addMarker(lat, lng);
            this.updateFormFields(lat, lng);
        } else {
            alert('Location is outside the Rinconada area. Please select a location within Iriga City, Baao, Buhi, Nabua, or Bato.');
        }
    }

    isWithinBounds(lat, lng) {
        return lat >= this.bounds[0][0] && lat <= this.bounds[1][0] &&
               lng >= this.bounds[0][1] && lng <= this.bounds[1][1];
    }

    getLocation() {
        return {
            latitude: this.latitude,
            longitude: this.longitude
        };
    }
}

// Global function to initialize location picker
function initLocationPicker(containerId, options = {}) {
    console.log('initLocationPicker called with containerId:', containerId, 'options:', options);
    try {
        return new LocationPicker(containerId, options);
    } catch (error) {
        console.error('Error initializing LocationPicker:', error);
        return null;
    }
}

// Utility function to get current location
function getCurrentLocation(callback) {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                callback(position.coords.latitude, position.coords.longitude);
            },
            (error) => {
                console.error('Geolocation error:', error);
                alert('Unable to get your current location. Please search for a location instead.');
            }
        );
    } else {
        alert('Geolocation is not supported by this browser.');
    }
}
