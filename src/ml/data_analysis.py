"""
Coral Health Data Analysis Module
Comprehensive analysis and visualization of coral health datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Plotly imports - will be imported only when needed
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import cv2
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoralDataAnalyzer:
    """
    Comprehensive coral health data analysis system
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.class_names = ['Dead', 'Healthy', 'Unhealthy']
        self.class_colors = {
            'Dead': '#8B4513',      # Brown
            'Healthy': '#32CD32',    # Lime Green
            'Unhealthy': '#FFD700'   # Gold
        }
    
    def analyze_dataset_distribution(self, data_summary: pd.DataFrame) -> Dict:
        """
        Analyze class distribution in the dataset
        
        Args:
            data_summary: DataFrame with class distribution data
            
        Returns:
            Analysis results dictionary
        """
        total_samples = data_summary['Count'].sum()
        
        # Calculate balance metrics
        class_counts = data_summary['Count'].values
        balance_ratio = np.max(class_counts) / np.min(class_counts)
        coefficient_variation = np.std(class_counts) / np.mean(class_counts)
        
        # Determine balance status
        if balance_ratio <= 1.5:
            balance_status = "Well-balanced"
        elif balance_ratio <= 2.0:
            balance_status = "Moderately imbalanced"
        else:
            balance_status = "Highly imbalanced"
        
        analysis = {
            'total_samples': int(total_samples),
            'class_distribution': data_summary.to_dict('records'),
            'balance_metrics': {
                'balance_ratio': float(balance_ratio),
                'coefficient_variation': float(coefficient_variation),
                'balance_status': balance_status
            },
            'recommendations': self._get_balance_recommendations(balance_ratio)
        }
        
        return analysis
    
    def _get_balance_recommendations(self, balance_ratio: float) -> List[str]:
        """Get recommendations based on class balance"""
        recommendations = []
        
        if balance_ratio > 2.0:
            recommendations.extend([
                "Consider data augmentation for minority classes",
                "Use stratified sampling for train/test splits",
                "Apply class weights during training",
                "Consider SMOTE or similar oversampling techniques"
            ])
        elif balance_ratio > 1.5:
            recommendations.extend([
                "Use stratified sampling for splits",
                "Monitor per-class performance metrics",
                "Consider weighted loss functions"
            ])
        else:
            recommendations.append("Dataset is well-balanced for training")
        
        return recommendations
    
    def analyze_color_characteristics(self, image_paths: List[str], 
                                   labels: List[str], 
                                   sample_size: int = 100) -> Dict:
        """
        Analyze color characteristics of coral images by class
        
        Args:
            image_paths: List of image file paths
            labels: Corresponding class labels
            sample_size: Number of images to sample per class
            
        Returns:
            Color analysis results
        """
        results = {}
        
        for class_name in self.class_names:
            # Get images for this class
            class_images = [
                path for path, label in zip(image_paths, labels) 
                if label == class_name
            ]
            
            # Sample images if too many
            if len(class_images) > sample_size:
                class_images = np.random.choice(class_images, sample_size, replace=False)
            
            # Analyze color properties
            color_stats = self._analyze_class_colors(class_images)
            results[class_name] = color_stats
        
        return results
    
    def _analyze_class_colors(self, image_paths: List[str]) -> Dict:
        """Analyze color properties for a specific class"""
        rgb_values = []
        brightness_values = []
        contrast_values = []
        
        for img_path in image_paths:
            try:
                # Load and process image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Calculate RGB statistics
                mean_rgb = np.mean(image_rgb, axis=(0, 1))
                rgb_values.append(mean_rgb)
                
                # Calculate brightness (luminance)
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Calculate contrast (standard deviation of grayscale)
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {e}")
                continue
        
        # Convert to numpy arrays
        rgb_values = np.array(rgb_values)
        brightness_values = np.array(brightness_values)
        contrast_values = np.array(contrast_values)
        
        # Calculate statistics
        stats = {
            'rgb_mean': np.mean(rgb_values, axis=0).tolist(),
            'rgb_std': np.std(rgb_values, axis=0).tolist(),
            'brightness_mean': float(np.mean(brightness_values)),
            'brightness_std': float(np.std(brightness_values)),
            'contrast_mean': float(np.mean(contrast_values)),
            'contrast_std': float(np.std(contrast_values)),
            'sample_count': len(rgb_values)
        }
        
        return stats
    
    def create_distribution_plots(self, data_summary: pd.DataFrame, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create class distribution visualization plots
        
        Args:
            data_summary: DataFrame with class distribution
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = [self.class_colors[cls] for cls in data_summary['Class']]
        wedges, texts, autotexts = axes[0].pie(
            data_summary['Percentage'], 
            labels=data_summary['Class'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=(0.05, 0.05, 0.05)
        )
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = axes[1].bar(
            data_summary['Class'], 
            data_summary['Count'],
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        axes[1].set_title('Number of Images per Class', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Coral Health Status')
        axes[1].set_ylabel('Number of Images')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        return fig
    
    def create_color_analysis_plots(self, color_stats: Dict, 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive color analysis visualizations
        
        Args:
            color_stats: Color statistics by class
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for plotting
        classes = list(color_stats.keys())
        rgb_means = [color_stats[cls]['rgb_mean'] for cls in classes]
        brightness_means = [color_stats[cls]['brightness_mean'] for cls in classes]
        brightness_stds = [color_stats[cls]['brightness_std'] for cls in classes]
        contrast_means = [color_stats[cls]['contrast_mean'] for cls in classes]
        contrast_stds = [color_stats[cls]['contrast_std'] for cls in classes]
        
        # 1. RGB Channel Comparison
        x_pos = np.arange(len(classes))
        width = 0.25
        
        rgb_data = np.array(rgb_means)
        r_vals, g_vals, b_vals = rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2]
        
        axes[0, 0].bar(x_pos - width, r_vals, width, label='R Channel', 
                      color='red', alpha=0.7)
        axes[0, 0].bar(x_pos, g_vals, width, label='G Channel', 
                      color='green', alpha=0.7)
        axes[0, 0].bar(x_pos + width, b_vals, width, label='B Channel', 
                      color='blue', alpha=0.7)
        
        axes[0, 0].set_title('Average RGB Values by Class', fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Average Channel Value')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(classes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Brightness Distribution
        axes[0, 1].bar(classes, brightness_means, 
                      yerr=brightness_stds,
                      color=[self.class_colors[cls] for cls in classes],
                      alpha=0.7, capsize=5)
        axes[0, 1].set_title('Brightness Distribution by Class', fontweight='bold')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Brightness (μ ± σ)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Contrast Distribution
        axes[1, 0].bar(classes, contrast_means,
                      yerr=contrast_stds,
                      color=[self.class_colors[cls] for cls in classes],
                      alpha=0.7, capsize=5)
        axes[1, 0].set_title('Contrast Distribution by Class', fontweight='bold')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Contrast (Standard Deviation)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Combined RGB Analysis (RGB as single metric)
        combined_rgb = [np.mean(rgb) for rgb in rgb_means]
        axes[1, 1].scatter(brightness_means, contrast_means, 
                          s=[rgb*3 for rgb in combined_rgb],
                          c=[self.class_colors[cls] for cls in classes],
                          alpha=0.7, edgecolors='black')
        
        for i, cls in enumerate(classes):
            axes[1, 1].annotate(cls, 
                               (brightness_means[i], contrast_means[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontweight='bold')
        
        axes[1, 1].set_title('Brightness vs Contrast Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Brightness')
        axes[1, 1].set_ylabel('Contrast')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Color analysis plot saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, data_summary: pd.DataFrame, 
                                   color_stats: Dict):
        """
        Create interactive Plotly dashboard
        
        Args:
            data_summary: Class distribution data
            color_stats: Color analysis statistics
            
        Returns:
            Plotly figure with subplots
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available. Skipping interactive dashboard.")
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Class Distribution', 
                'RGB Channel Analysis',
                'Brightness vs Contrast',
                'Sample Statistics'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Pie chart for class distribution
        fig.add_trace(
            go.Pie(
                labels=data_summary['Class'],
                values=data_summary['Count'],
                marker_colors=[self.class_colors[cls] for cls in data_summary['Class']],
                hovertemplate="<b>%{label}</b><br>" +
                             "Count: %{value}<br>" +
                             "Percentage: %{percent}<br>" +
                             "<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. RGB channel analysis
        classes = list(color_stats.keys())
        rgb_means = np.array([color_stats[cls]['rgb_mean'] for cls in classes])
        
        for i, channel in enumerate(['R', 'G', 'B']):
            fig.add_trace(
                go.Bar(
                    name=f'{channel} Channel',
                    x=classes,
                    y=rgb_means[:, i],
                    marker_color=['red', 'green', 'blue'][i],
                    opacity=0.7,
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Brightness vs Contrast scatter
        brightness_vals = [color_stats[cls]['brightness_mean'] for cls in classes]
        contrast_vals = [color_stats[cls]['contrast_mean'] for cls in classes]
        
        fig.add_trace(
            go.Scatter(
                x=brightness_vals,
                y=contrast_vals,
                mode='markers+text',
                text=classes,
                textposition="top center",
                marker=dict(
                    size=20,
                    color=[self.class_colors[cls] for cls in classes],
                    line=dict(width=2, color='black')
                ),
                showlegend=False,
                hovertemplate="<b>%{text}</b><br>" +
                             "Brightness: %{x:.2f}<br>" +
                             "Contrast: %{y:.2f}<br>" +
                             "<extra></extra>"
            ),
            row=2, col=1
        )
        
        # 4. Statistics table
        table_data = []
        for cls in classes:
            stats = color_stats[cls]
            table_data.append([
                cls,
                f"{stats['brightness_mean']:.1f}±{stats['brightness_std']:.1f}",
                f"{stats['contrast_mean']:.1f}±{stats['contrast_std']:.1f}",
                str(stats['sample_count'])
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Class', 'Brightness (μ±σ)', 'Contrast (μ±σ)', 'Samples'],
                    fill_color='lightblue',
                    font=dict(size=12, color='black'),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lightgray',
                    font=dict(size=11),
                    align='center'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Coral Health Dataset Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_analysis_report(self, data_summary: pd.DataFrame,
                               color_stats: Dict,
                               output_dir: str = "reports") -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            data_summary: Class distribution data
            color_stats: Color analysis statistics
            output_dir: Directory to save the report
            
        Returns:
            Path to generated report
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Analyze dataset distribution
        dist_analysis = self.analyze_dataset_distribution(data_summary)
        
        # Generate report content
        report_content = self._create_report_content(dist_analysis, color_stats)
        
        # Save report
        report_path = os.path.join(output_dir, "coral_health_analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Analysis report generated: {report_path}")
        return report_path
    
    def _create_report_content(self, dist_analysis: Dict, color_stats: Dict) -> str:
        """Create formatted report content"""
        
        from datetime import datetime
        
        report = f"""# Coral Health Classification Dataset Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
**Total Images**: {dist_analysis['total_samples']:,}
**Classes**: {len(self.class_names)} ({', '.join(self.class_names)})

## Class Distribution
"""
        
        for item in dist_analysis['class_distribution']:
            report += f"- **{item['Class']}**: {item['Count']} images ({item['Percentage']:.1f}%)\n"
        
        report += f"""
**Balance Analysis**: 
- Class imbalance ratio: {dist_analysis['balance_metrics']['balance_ratio']:.2f}:1
- Coefficient of variation: {dist_analysis['balance_metrics']['coefficient_variation']:.3f}
- Balance score: {dist_analysis['balance_metrics']['balance_status']}

## Visual Characteristics by Class

"""
        
        for class_name in self.class_names:
            if class_name in color_stats:
                stats = color_stats[class_name]
                report += f"""### {class_name} Coral
- Brightness: {stats['brightness_mean']:.1f} ± {stats['brightness_std']:.1f}
- Contrast: {stats['contrast_mean']:.1f} ± {stats['contrast_std']:.1f}
- RGB Mean: ({stats['rgb_mean'][0]:.1f}, {stats['rgb_mean'][1]:.1f}, {stats['rgb_mean'][2]:.1f})

"""
        
        report += f"""## Machine Learning Suitability

### Strengths
✓ Clear visual distinctions between classes
✓ Adequate sample sizes for deep learning
✓ {dist_analysis['balance_metrics']['balance_status'].lower()} class distribution
✓ Consistent image format and size
✓ Color-based features show good separability

### Recommendations
"""
        
        for rec in dist_analysis['recommendations']:
            report += f"- {rec}\n"
        
        report += """
### Model Selection Suggestions
1. **CNN Architectures**: EfficientNet, ResNet, or MobileNet for image classification
2. **Transfer Learning**: Use pre-trained ImageNet models as base
3. **Data Augmentation**: Rotation, flip, brightness/contrast adjustments
4. **Evaluation**: Use stratified cross-validation and per-class metrics

## Conclusion
The coral health classification dataset provides a solid foundation for developing 
robust machine learning models for coral reef monitoring and conservation efforts.
"""
        
        return report


# Utility functions
def load_coral_dataset_info(csv_path: str) -> pd.DataFrame:
    """Load coral dataset summary from CSV"""
    return pd.read_csv(csv_path)

def extract_sample_images(image_dir: str, num_samples: int = 4) -> Dict[str, List[str]]:
    """Extract sample images for each class"""
    samples = {}
    
    for class_name in ['Dead', 'Healthy', 'Unhealthy']:
        class_dir = os.path.join(image_dir, class_name.lower())
        if os.path.exists(class_dir):
            image_files = [
                f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if len(image_files) >= num_samples:
                selected = np.random.choice(image_files, num_samples, replace=False)
                samples[class_name] = [
                    os.path.join(class_dir, f) for f in selected
                ]
            else:
                samples[class_name] = [
                    os.path.join(class_dir, f) for f in image_files
                ]
    
    return samples


if __name__ == "__main__":
    # Example usage
    analyzer = CoralDataAnalyzer()
    
    # Load dataset summary
    data_summary = pd.DataFrame({
        'Class': ['Healthy', 'Unhealthy', 'Dead'],
        'Count': [661, 508, 430],
        'Percentage': [41.3, 31.8, 26.9],
        'Description': [
            'Vibrant, colorful coral with good health',
            'Bleached or stressed coral, whitish/pale',
            'Dead coral, darker/brownish, no living tissue'
        ]
    })
    
    # Analyze distribution
    dist_analysis = analyzer.analyze_dataset_distribution(data_summary)
    print("Dataset Analysis:", json.dumps(dist_analysis, indent=2))
    
    # Create visualizations
    fig = analyzer.create_distribution_plots(data_summary)
    plt.show()