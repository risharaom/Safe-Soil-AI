import { GoogleMap, Marker, useJsApiLoader } from '@react-google-maps/api';

const mapContainerStyle = {
  width: '100%',
  height: '400px'
};
const center = {
  lat: 37.8044,
  lng: -122.2712
};

function MapWithAMarker() {
  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: import.meta.env.VITE_APP_GOOGLE_MAPS_API_KEY
  });

  return isLoaded ? (
    <GoogleMap mapContainerStyle={mapContainerStyle} zoom={8} center={center}>
      <Marker position={center} />
    </GoogleMap>
  ) : (
    <></>
  );
}

export default MapWithAMarker;
