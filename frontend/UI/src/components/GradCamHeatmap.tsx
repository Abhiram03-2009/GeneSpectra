// frontend/UI/src/components/GradCamHeatmap.tsx
import React from 'react';
import { motion } from 'framer-motion';

// This component now just receives the visualization images
interface GradCamHeatmapProps {
  visualizations: { panel_a: string; panel_b: string; panel_c: string; confusion_matrix: string; letter_encoded_output: string; base_specific_activations: string; };
}

// Reusable ZoomableImage Component
const ZoomableImage: React.FC<{ src: string; alt: string; title: string }> = ({ src, alt, title }) => {
  const [scale, setScale] = React.useState(1);
  const [translateX, setTranslateX] = React.useState(0);
  const [translateY, setTranslateY] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const [startX, setStartX] = React.useState(0);
  const [startY, setStartY] = React.useState(0);

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const scaleAmount = 0.1;
    const newScale = e.deltaY < 0 ? scale * (1 + scaleAmount) : scale / (1 + scaleAmount);
    setScale(Math.max(0.5, Math.min(5, newScale))); // Limit zoom between 0.5x and 5x
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
    setStartX(e.clientX);
    setStartY(e.clientY);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    setTranslateX(prev => prev + (e.clientX - startX));
    setTranslateY(prev => prev + (e.clientY - startY));
    setStartX(e.clientX);
    setStartY(e.clientY);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div 
      className="relative w-full h-full overflow-hidden cursor-grab active:cursor-grabbing rounded-md border border-gray-700"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp} // Stop dragging if mouse leaves the element
    >
      <h4 className="absolute top-2 left-2 text-cyan-300 font-semibold text-sm bg-black/50 px-2 py-1 rounded-md z-10">{title}</h4>
      <img 
        src={src}
        alt={alt}
        className="w-full h-full object-contain absolute top-0 left-0"
        style={{
          transform: `scale(${scale}) translate(${translateX / scale}px, ${translateY / scale}px)`,
          transformOrigin: '0 0',
          transition: isDragging ? 'none' : 'transform 0.1s ease-out',
        }}
      />
    </div>
  );
};

const GradCamHeatmap: React.FC<GradCamHeatmapProps> = ({ visualizations }) => {
  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ staggerChildren: 0.1 }}
      className="w-full h-full flex flex-col p-4 bg-gradient-to-br from-[#0c1a2e] to-[#1a2e4b] rounded-xl shadow-2xl border border-cyan-700/50"
    >
      <h2 className="text-3xl font-bold text-center text-cyan-300 mb-6 flex-shrink-0 animate-pulse">
        Deep Dive Visualizations
      </h2>
      <div className="flex-grow grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 overflow-y-auto custom-scrollbar p-2">
        {/* Panel A: Original 2D DNA Sequence Structure */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg relative overflow-hidden">
            {visualizations.panel_a && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.panel_a}`} 
              alt="Original DNA Sequence" 
              title="A. Original 2D DNA Sequence" 
            />}
        </motion.div>
        
        {/* Panel B: Grad-CAM Activation Heatmap */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg relative overflow-hidden">
            {visualizations.panel_b && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.panel_b}`} 
              alt="Grad-CAM Heatmap" 
              title="B. Grad-CAM Activation Heatmap" 
            />}
        </motion.div>
        
        {/* Panel C: DNA + Grad-CAM Overlay */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg relative overflow-hidden">
            {visualizations.panel_c && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.panel_c}`} 
              alt="DNA + Grad-CAM Overlay" 
              title="C. DNA + Grad-CAM Overlay" 
            />}
        </motion.div>

        {/* Confusion Matrix */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg relative overflow-hidden">
            {visualizations.confusion_matrix && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.confusion_matrix}`} 
              alt="Confusion Matrix" 
              title="Confusion Matrix" 
            />}
        </motion.div>
        
        {/* Base-Specific Activations Box Plots */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg relative overflow-hidden">
            {visualizations.base_specific_activations && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.base_specific_activations}`} 
              alt="Base-Specific Activations Box Plots" 
              title="E. Base-Specific Activations" 
            />}
        </motion.div>

        {/* Letter Encoded Output */}
        <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="bg-[#0c1a2e] p-4 rounded-lg flex flex-col items-center justify-center border border-cyan-700/50 shadow-lg lg:col-span-3 relative overflow-hidden">
            {visualizations.letter_encoded_output && <ZoomableImage 
              src={`data:image/png;base64,${visualizations.letter_encoded_output}`} 
              alt="Letter Encoded Output" 
              title="Letter Encoded Output" 
            />}
        </motion.div>
      </div>
    </motion.div>
  );
};

export default GradCamHeatmap;