// PlantDiseaseDetection.tsx
import React, { useState } from 'react';
import { View, Button, Image, Text, StyleSheet, ScrollView, Animated, TouchableOpacity } from 'react-native';
import { launchImageLibrary, ImagePickerResponse } from 'react-native-image-picker';
import axios from 'axios';

const PlantDiseaseDetection = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageData, setImageData] = useState<any | null>(null);
  const [predictedClass, setPredictedClass] = useState<string | null>(null);
  const [fadeAnim] = useState(new Animated.Value(0)); // For animation

  const cleanText = (text: string) => {
    return text.replace(/[*#]/g, '');
  };

  const selectImage = async () => {
    const response: ImagePickerResponse = await launchImageLibrary({ mediaType: 'photo' });
    if (!response.didCancel && response.assets && response.assets[0]) {
      setImageUri(response.assets[0].uri || null);
      setImageData(response.assets[0]);
    }
  };

  const uploadImage = async () => {
    if (!imageData) {
      console.error('No image selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', {
      uri: imageData.uri,
      type: imageData.type,
      name: imageData.fileName || 'image.jpg',
    });

    try {
      const uploadResponse = await axios.post('http://10.0.2.2:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (uploadResponse.data.prediction) {
        const cleanedPrediction = cleanText(uploadResponse.data.prediction);
        setPredictedClass(cleanedPrediction);
        fadeIn(); // Trigger fade-in animation
      }
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  // Animation for fade-in effect
  const fadeIn = () => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
  };

  return (
    <View style={styles.container}>
      {/* Header Section */}
      <View style={styles.header}>
        <Text style={styles.headerText}>Plant Disease Detection System</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <Text style={styles.title}>Plant Disease Detection and Cure</Text>
        <Text style={styles.subtitle}>Upload an image of your plant to detect any diseases and get recommended treatments.</Text>

        <TouchableOpacity style={styles.button} onPress={selectImage}>
          <Text style={styles.buttonText}>Select Image</Text>
        </TouchableOpacity>
        {imageUri && (
          <>
            <Text style={styles.selectedImageLabel}>Selected Image:</Text>
            <Image source={{ uri: imageUri }} style={styles.image} />
            <TouchableOpacity style={styles.button} onPress={uploadImage}>
              <Text style={styles.buttonText}>Upload Image</Text>
            </TouchableOpacity>
          </>
        )}
        {predictedClass && (
          <Animated.View style={[styles.predictionContainer, { opacity: fadeAnim }]}>
            <Text style={styles.predictionText}>{predictedClass}</Text>
          </Animated.View>
        )}
      </ScrollView>

      {/* Footer Section */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>© 2024 Plant Disease Detection System</Text>
        <Text style={styles.footerText}>For more information, visit our website or contact support.</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#e1f5fe',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    width: '100%',
    backgroundColor: '#1e88e5',
    padding: 15,
    alignItems: 'center',
    borderBottomWidth: 2,
    borderColor: '#ffffff',
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginVertical: 20,
    color: '#1e88e5',
    textAlign: 'center',
    textShadowColor: '#000',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 20,
    color: '#455a64',
    textAlign: 'center',
    paddingHorizontal: 10,
  },
  selectedImageLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    marginVertical: 10,
    color: '#333',
  },
  image: {
    width: 250,
    height: 250,
    margin: 10,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#1e88e5',
  },
  predictionContainer: {
    marginVertical: 20,
    padding: 10,
    borderWidth: 1,
    borderColor: '#1e88e5',
    borderRadius: 8,
    backgroundColor: '#ffffff',
    width: '100%',
    alignItems: 'flex-start',
  },
  predictionText: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#1e88e5',
    textAlign: 'left',
    width: '100%',
  },
  button: {
    backgroundColor: '#6200ea',
    borderRadius: 5,
    paddingVertical: 12,
    paddingHorizontal: 20,
    marginVertical: 10,
    elevation: 3, // Shadow effect for Android
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  footer: {
    width: '100%',
    backgroundColor: '#1e88e5',
    padding: 10,
    alignItems: 'center',
    borderTopWidth: 2,
    borderColor: '#ffffff',
  },
  footerText: {
    fontSize: 14,
    color: '#ffffff',
    textAlign: 'center',
  },
});

export default PlantDiseaseDetection;
