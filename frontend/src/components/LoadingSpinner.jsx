// filepath: c:\Users\ifemi\OneDrive\Desktop\desd_group\DesD_AI_pathway\frontend\src\components\LoadingSpinner.jsx
import React from 'react';

const LoadingSpinner = ({ size = 'md', className = '' }) => {
  // Determine size classes
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4'
  }[size] || 'w-8 h-8 border-3';
  
  return (
    <div className={`animate-spin rounded-full ${sizeClasses} border-t-transparent ${className}`}
         style={{ borderTopColor: 'transparent', borderStyle: 'solid' }}
         role="status" 
         aria-label="Loading">
      <span className="sr-only">Loading...</span>
    </div>
  );
};

export default LoadingSpinner;