import React from 'react';
import { Row, Col, Card } from 'react-bootstrap';
import InfoBox from './google-maps/InfoBox';
import MarkerClusterer from './google-maps/MarkerClusterer';
import Marker from './google-maps/Marker';
import StreetViewPanorma from './google-maps/StreetViewPanorma';

const handlePredict = async () => {
  const lat = parseFloat(document.getElementById("Latitude").value);
  const lon = parseFloat(document.getElementById("Longitude").value);
  console.log(lat,lon)

  if (isNaN(lat) || isNaN(lon)) {
    alert("Please enter valid latitude and longitude.");
    return;
  }

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ latitude: lat, longitude: lon })
    });

    // Always try to parse response as JSON
    const data = await response.json();

    if (response.ok && data.prediction !== undefined) {
      alert(`Predicted Lead Concentration: ${data.prediction} ppm`);
    } else {
      alert("Prediction failed: " + (data.error || "Unknown error"));
    }
  } catch (err) {
    alert("Network or server error: " + err.message);
  }
};

const GoogleMaps = () => {
  return (
    <React.Fragment>
      <Row>
        <Col xl={9} style={{ position: 'relative' }}>
          <Card>
            <Card.Header>
              <Card.Title as="h5">Lead Concentration</Card.Title>
              <div>
                <p></p></div>
              <Card.Title as="h6"> Only have data for Oakland and Peidmont Areas</Card.Title>
            </Card.Header>
            <Card.Body>
              <div style={{ width: '100%', height: '500px', position: 'relative' }}>
                {/* Map focused on Bay Area */}
                <Marker />
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col xl={3} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
          <Card style={{ width: '100%' }}>
            <Card.Body>
              <button
              //onClick={handlePredict}
                style={{
                  width: '180px',
                  borderRadius: '8px',
                  backgroundColor: "#2c3e50", // Green color for a professional look
                  color: 'white',
                  border: 'none',
                  padding: '14px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontFamily: "Arial, sans-serif", // Professional font
                  marginBottom: '20px', // Space between button and inputs
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                  transition: 'background-color 0.3s ease, transform 0.2s ease',
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = '#2c3e50'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#2c3e50'}
                onClick={handlePredict}
                //onClick={() => alert('Predict button clicked')}
              >
                Predict
              </button>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <input
                  type="number"
                  //step="any"
                  id="Latitude"
                  placeholder="Latitude"
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid #ccc',
                    fontSize: '14px',
                    fontFamily: 'Roboto, Helvetica, Arial, sans-serif',
                    marginBottom: '10px',
                  }}
                />
                <input
                  type="number"
                  //step="any"
                  id="Longitude"
                  placeholder="Longitude"
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid #ccc',
                    fontSize: '14px',
                    fontFamily: 'Roboto, Helvetica, Arial, sans-serif',
                  }}
                />
              </div>
              
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </React.Fragment>
  );
};
const otherMaps = ()=>{
  return(
    <React.Fragment>
      <Row>
      <Col xl={6}>
          <Card>
            <Card.Header>
              <Card.Title as="h5">Infobox</Card.Title>
            </Card.Header>
            <Card.Body>
              <InfoBox />
            </Card.Body>
          </Card>
        </Col>
        <Col xl={6}>
        
          <Card>
            <Card.Header>
              <Card.Title as="h5">Marker Clusterer</Card.Title>
            </Card.Header>
            <Card.Body>
              <MarkerClusterer />
            </Card.Body>
          </Card>
        </Col>
        <Col xl={6}>
          <Card>
            <Card.Header>
              <Card.Title as="h5">Street View Panorma</Card.Title>
            </Card.Header>
            <Card.Body>
              <StreetViewPanorma />
            </Card.Body>
          </Card>
        </Col>

      </Row>
    </React.Fragment>


  );
};

export default GoogleMaps;
