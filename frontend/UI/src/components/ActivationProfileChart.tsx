import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Label } from 'recharts';

interface ChartData {
  activationProfile: any[];
  baseActivations: any;
}

interface ActivationProfileChartProps {
  chartData: ChartData;
}

const ActivationProfileChart: React.FC<ActivationProfileChartProps> = ({ chartData }) => {
  const chartDataToRender = chartData; // Ensure we're using the actual data

  return (
    <section className="hud-card relative overflow-hidden rounded-3xl border border-cyan-600/40 backdrop-blur-lg bg-gradient-to-br from-[#0a1e33]/90 to-[#142b43]/90 shadow-2xl p-8 hover:border-cyan-300 transition-all duration-300 group w-full">
      <h3 className="text-2xl font-bold text-white mb-6 text-center tracking-tight">Activation Profile</h3>
      <ResponsiveContainer width="100%" height={470} >
        <AreaChart data={chartDataToRender.activationProfile} margin={{ top: 10, right: 30, left: 0, bottom: 0 }} >
          <XAxis dataKey="position" stroke="#39FF14" tick={{ fill: '#39FF14', fontSize: '10px' }}> 
            <Label value="Position" offset={0} position="insideBottom" fill="#39FF14" fontSize={13} />
          </XAxis>
          <YAxis stroke="#39FF14" domain={[0, 1]} tick={{ fill: '#39FF14', fontSize: '10px' }}> 
            <Label value="Activation" angle={-90} position="insideLeft" offset={10} fill="#39FF14" fontSize={13} />
          </YAxis>
          <Tooltip contentStyle={{ backgroundColor: '#0a1e33', borderColor: '#0ea5e9', color: '#fff' }} />
          <Area 
            type="monotone" 
            dataKey="activation" 
            stroke="#0ea5e9" 
            fill="#0ea5e9"   
            strokeOpacity={1}
            fillOpacity={0.6}
          />
        </AreaChart>
      </ResponsiveContainer>
    </section>
  );
};

export default ActivationProfileChart;
