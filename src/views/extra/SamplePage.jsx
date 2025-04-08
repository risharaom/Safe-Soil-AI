import React from 'react';
import { Row, Col } from 'react-bootstrap';

import Card from '../../components/Card/MainCard';
//import userImage from '../src/assets/images/user/chart.png';
import userImage from '../../assets/images/user/chart.png'
import userImage2 from '../../assets/images/user/chart2.png'

const SamplePage = () => {
  return (
    <React.Fragment>
    <Row>
      <Col>
        <Card title="Graphs" isOption>
        <div className="mt-2" style={{ paddingBottom: '50px' }}>
              <a href="https://docs.google.com/document/d/1w_MfUSIfahyc2jbSR4qXe2gcuFrrSXwOtHy6EVhnSaI/edit?usp=sharing" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
              Learn about lead exposure in Bay Area communities
              </a>
            </div>
            
          <div className="flex justify-center gap-6 mb-4">
            {/* Graph/Image 1 with subheading */}
            <div className="flex flex-col items-center">
              <img
                src={userImage}
                alt="Graph 1"
                style={{ width: '800px', height: '400px' }}
                className="w-[200px] h-[100px] object-contain rounded"
                
                  // Width: 200px, Height: 100px
              />
              <p className="mt-2 text-sm font-medium text-gray-700 text-center">Peidmont - Lead exposure per hour vs Coverage</p>
            </div>
  
            {/* Graph/Image 2 with subheading */}
            <div className="flex flex-col items-center">
              <img
                src={userImage2}
                alt="Graph 2"
                style={{ width: '800px', height: '400px' }}
                className="w-[200px] h-[100px] object-contain rounded"  // Width: 200px, Height: 100px
              />
              <p className="mt-2 text-sm font-medium text-gray-700 text-center">West Oakland - Lead exposure per hour vs Coverage</p>
            </div>
          </div>
        </Card>
      </Col>
    </Row>
  </React.Fragment>
  );
};

export default SamplePage;
